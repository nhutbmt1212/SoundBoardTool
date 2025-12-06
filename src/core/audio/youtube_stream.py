"""YouTube Streaming - Stream YouTube audio to VB-Cable"""
import threading
import subprocess
import os

try:
    import sounddevice as sd
    import numpy as np
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    sd = None
    np = None

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    yt_dlp = None


def find_ffmpeg():
    """Find ffmpeg executable"""
    import shutil
    
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    
    common_paths = [
        os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\WinGet\Packages'),
        r'C:\ffmpeg\bin',
        r'C:\Program Files\ffmpeg\bin',
        os.path.expandvars(r'%USERPROFILE%\ffmpeg\bin'),
    ]
    
    for base in common_paths:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if 'ffmpeg.exe' in files:
                    return os.path.join(root, 'ffmpeg.exe')
    
    return None


FFMPEG_PATH = find_ffmpeg()


class YouTubeStream:
    """Handles YouTube audio streaming to VB-Cable"""
    
    def __init__(self, vb_manager):
        self.vb_manager = vb_manager
        self.volume = 1.0
        self.playing = False
        self.current_url = None
        self.current_title = None
        
        self._process = None
        self._thread = None
        self._stop_event = threading.Event()
    
    def play(self, url: str) -> dict:
        """Stream YouTube audio to VB-Cable"""
        if not YTDLP_AVAILABLE:
            return {'success': False, 'error': 'yt-dlp not installed'}
        
        if not SD_AVAILABLE or not self.vb_manager.is_connected():
            return {'success': False, 'error': 'VB-Cable not available'}
        
        self.stop()
        
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    return {'success': False, 'error': 'Cannot get video info'}
                
                title = info.get('title', 'Unknown')
                audio_url = self._extract_audio_url(info)
                
                if not audio_url:
                    return {'success': False, 'error': 'No audio stream found'}
                
                self.current_url = url
                self.current_title = title
                self._stop_event.clear()
                
                self._thread = threading.Thread(
                    target=self._stream_loop,
                    args=(audio_url,),
                    daemon=True
                )
                self._thread.start()
                self.playing = True
                
                print(f"▶ YouTube: {title}")
                return {'success': True, 'title': title}
                
        except Exception as e:
            print(f"YouTube error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_audio_url(self, info: dict) -> str:
        """Extract audio URL from yt-dlp info"""
        audio_url = info.get('url')
        
        if not audio_url:
            for fmt in info.get('formats', []):
                if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                    return fmt.get('url')
            
            for fmt in info.get('formats', []):
                if fmt.get('acodec') != 'none':
                    return fmt.get('url')
        
        return audio_url
    
    def _stream_loop(self, audio_url: str):
        """Background thread to stream YouTube audio via ffmpeg"""
        try:
            ffmpeg_exe = FFMPEG_PATH or 'ffmpeg'
            samplerate = self.vb_manager.get_samplerate()
            channels = self.vb_manager.get_channels()
            blocksize = 8192
            bytes_per_sample = 2 * channels
            
            # Start ffmpeg
            ffmpeg_cmd = [
                ffmpeg_exe,
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '5',
                '-i', audio_url,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', str(samplerate),
                '-ac', str(channels),
                '-'
            ]
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            self._process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=8192,
                startupinfo=startupinfo
            )
            
            import time
            time.sleep(0.5)
            
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode('utf-8', errors='ignore')
                print(f"FFmpeg error: {stderr[:500]}")
                return
            
            # Open streams
            vb_stream = sd.OutputStream(
                device=self.vb_manager.device_id,
                samplerate=samplerate,
                channels=channels,
                dtype='float32',
                blocksize=blocksize,
                latency='high',
                clip_off=True
            )
            vb_stream.start()
            
            speaker_stream = self._open_speaker_stream(samplerate, channels, blocksize)
            
            try:
                frame_count = 0
                while not self._stop_event.is_set():
                    raw_data = self._process.stdout.read(blocksize * bytes_per_sample)
                    if not raw_data:
                        break
                    
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    audio = audio.reshape(-1, channels)
                    audio *= self.volume
                    audio_out = np.ascontiguousarray(audio)
                    
                    vb_stream.write(audio_out)
                    
                    if speaker_stream:
                        try:
                            speaker_stream.write(audio_out)
                        except Exception:
                            pass
                    
                    frame_count += 1
                    if frame_count == 10:
                        level = np.abs(audio).max()
                        print(f"YouTube streaming... (audio level: {level:.4f})")
            finally:
                vb_stream.stop()
                vb_stream.close()
                if speaker_stream:
                    speaker_stream.stop()
                    speaker_stream.close()
                    
        except Exception as e:
            if not self._stop_event.is_set():
                print(f"YouTube stream error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._cleanup_process()
            self.playing = False
            print("YouTube stream ended")
    
    def _open_speaker_stream(self, samplerate, channels, blocksize):
        """Open speaker stream for monitoring"""
        try:
            default_output = sd.default.device[1]
            if default_output is not None and default_output != self.vb_manager.device_id:
                stream = sd.OutputStream(
                    device=default_output,
                    samplerate=samplerate,
                    channels=channels,
                    dtype='float32',
                    blocksize=blocksize,
                    latency='high',
                    clip_off=True
                )
                stream.start()
                return stream
        except Exception as e:
            print(f"Speaker output not available: {e}")
        return None
    
    def _cleanup_process(self):
        """Clean up ffmpeg process"""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
    
    def stop(self):
        """Stop YouTube streaming"""
        self._stop_event.set()
        self._cleanup_process()
        
        if self._thread:
            try:
                self._thread.join(timeout=2)
            except Exception:
                pass
            self._thread = None
        
        self.playing = False
        self.current_url = None
        self.current_title = None
    
    def is_playing(self) -> bool:
        return self.playing
    
    def get_info(self) -> dict:
        """Get current stream info"""
        return {
            'playing': self.playing,
            'title': self.current_title,
            'url': self.current_url
        }
    
    def set_volume(self, vol: float):
        """Set stream volume (0.0 - 2.0)"""
        self.volume = max(0.0, min(2.0, vol))
