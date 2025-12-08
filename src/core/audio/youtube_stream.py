"""YouTube Streaming - Stream YouTube audio to VB-Cable"""
import threading
import subprocess
import os
import json
import hashlib
import sys
from pathlib import Path

# Import AppData path helper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config_paths import get_youtube_cache_dir

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
    """Handles YouTube audio streaming to VB-Cable with persistent cache"""
    
    def __init__(self, vb_manager):
        self.vb_manager = vb_manager
        self.volume = 1.0
        self.pitch = 1.0
        self.trim_start = 0.0
        self.trim_end = 0.0
        self.playing = False
        self.paused = False
        self.current_url = None
        self.current_title = None
        
        self._process = None
        self._thread = None
        self._stop_event = threading.Event()
        
        # Cache directory - use AppData for proper permissions
        self.cache_dir = Path(get_youtube_cache_dir())
        self.cache_index_file = self.cache_dir / 'index.json'
        self._cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> dict:
        """Load cache index từ file"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_cache_index(self):
        """Save cache index ra file"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save cache index: {e}")
    
    def _get_cache_key(self, url: str) -> str:
        """Tạo cache key từ URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached_file(self, url: str) -> tuple:
        """Kiểm tra xem URL đã được cache chưa. Return (filepath, title) hoặc (None, None)"""
        cache_key = self._get_cache_key(url)
        if cache_key in self._cache_index:
            cached = self._cache_index[cache_key]
            filepath = Path(cached['file'])
            
            # Check if file exists
            if filepath.exists():
                # If file has no extension, try to detect and rename it
                if not filepath.suffix:
                    new_path = self._fix_cached_file_extension(filepath)
                    if new_path:
                        # Update cache index
                        self._cache_index[cache_key]['file'] = str(new_path)
                        self._save_cache_index()
                        return str(new_path), cached['title']
                
                return str(filepath), cached['title']
        return None, None
    
    def _fix_cached_file_extension(self, filepath: Path) -> Path:
        """Try to detect file format and add proper extension"""
        try:
            import shutil
            # Try common audio extensions (m4a is most common from YouTube)
            for ext in ['.m4a', '.webm', '.opus', '.mp3']:
                new_path = filepath.with_suffix(ext)
                shutil.move(str(filepath), str(new_path))
                return new_path
        except Exception:
            pass
        
        return None
    
    def _download_and_cache(self, url: str, progress_callback=None) -> tuple:
        """Tải video về cache. Return (filepath, title)"""
        cache_key = self._get_cache_key(url)
        output_template = self.cache_dir / f"{cache_key}.%(ext)s"
        
        hooks = []
        if progress_callback:
            hooks.append(progress_callback)
            
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': str(output_template),  # Use %(ext)s to get extension
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': hooks,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                
                # Tìm file đã tải (yt-dlp sẽ thêm extension)
                output_file = None
                for ext in ['.m4a', '.webm', '.opus', '.mp3', '.mp4', '']:
                    test_file = self.cache_dir / f"{cache_key}{ext}"
                    if test_file.exists():
                        output_file = test_file
                        break
                
                if not output_file:
                    return None, None
                
                # Lưu vào index
                self._cache_index[cache_key] = {
                    'url': url,
                    'title': title,
                    'file': str(output_file)
                }
                self._save_cache_index()
                
                return str(output_file), title
                
        except Exception as e:
            print(f"Download error: {e}")
            return None, None
    
    def play(self, url: str, progress_callback=None) -> dict:
        """Stream YouTube audio to VB-Cable (with persistent cache)"""
        if not YTDLP_AVAILABLE:
            return {'success': False, 'error': 'yt-dlp not installed'}
        
        if not SD_AVAILABLE or not self.vb_manager.is_connected():
            return {'success': False, 'error': 'VB-Cable not available'}
        
        self.stop()
        
        try:
            # Kiểm tra cache trước - nếu có thì phát ngay (GẦN NHƯ TỨC THÌ!)
            cached_file, title = self._get_cached_file(url)
            
            if cached_file:
                audio_source = cached_file
            else:
                # Chưa có cache - tải về lần đầu
                cached_file, title = self._download_and_cache(url, progress_callback)
                if not cached_file:
                    return {'success': False, 'error': 'Failed to download'}
                audio_source = cached_file
            
            # Phát file đã cache
            self.current_url = url
            self.current_title = title
            self.paused = False
            self._stop_event.clear()
            
            self._thread = threading.Thread(
                target=self._stream_loop,
                args=(audio_source,),
                daemon=True
            )
            self._thread.start()
            self.playing = True
            
            return {'success': True, 'title': title}
                
        except Exception as e:
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
    
    def _stream_loop(self, audio_source: str):
        """Background thread to stream audio via ffmpeg (from file or URL)"""
        try:
            ffmpeg_exe = FFMPEG_PATH or 'ffmpeg'
            samplerate = self.vb_manager.get_samplerate()
            channels = self.vb_manager.get_channels()
            blocksize = 8192
            bytes_per_sample = 2 * channels
            
            # Kiểm tra xem là file local hay URL
            is_local_file = os.path.exists(audio_source)
            
            # Build FFmpeg command with filtering support
            filter_complex = []
            
            # Volume is handled in python now, but let's keep it simple
            # Pitch shifting
            if abs(self.pitch - 1.0) > 0.01:
                # asetrate method for simple resampling pitch shift (chipmunk effect)
                # new_rate = sample_rate * pitch
                new_rate = int(samplerate * self.pitch)
                filter_complex.append(f"asetrate={new_rate}")
                
            # If we need to maintain standard sample rate output after asetrate,
            # we might need resampling, but playing raw PCM usually dictates the rate
            # However, VB-Cable expects specific rate. 
            # If we change rate, we speed up/slow down.
            # Wait, standard chipmunk is speed up.
            
            # Construct command
            cmd = [ffmpeg_exe]
            
            if not is_local_file:
                cmd.extend([
                    '-reconnect', '1',
                    '-reconnect_streamed', '1',
                    '-reconnect_delay_max', '5'
                ])
            
            # Add trim parameters if set
            if self.trim_start > 0:
                cmd.extend(['-ss', str(self.trim_start)])
                
            cmd.extend(['-i', audio_source])
            
            if self.trim_end > 0:
                duration = self.trim_end - self.trim_start
                cmd.extend(['-t', str(duration)])
            
            if filter_complex:
                cmd.extend(['-af', ','.join(filter_complex)])
                
            cmd.extend([
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', str(samplerate),
                '-ac', str(channels),
                '-'
            ])
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=8192,
                startupinfo=startupinfo
            )
            
            import time
            time.sleep(0.5)
            
            if self._process.poll() is not None:
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
                    # Handle pause
                    if self.paused:
                        import time
                        time.sleep(0.1)
                        continue

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
            finally:
                vb_stream.stop()
                vb_stream.close()
                if speaker_stream:
                    speaker_stream.stop()
                    speaker_stream.close()
                    
        except Exception as e:
            if not self._stop_event.is_set():
                pass
        finally:
            self._cleanup_process()
            self.playing = False
    
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
            pass
        return None
    
    def _cleanup_process(self):
        """Clean up ffmpeg process"""
        if self._process:
            try:
                from core.utils import kill_process_tree
                kill_process_tree(self._process.pid)
            except Exception:
                # Fallback
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
    
    def stop(self):
        """Stop YouTube streaming"""
        # Signal stop first
        self._stop_event.set()
        
        # Clean up process
        self._cleanup_process()
        
        # Wait for thread to finish with increased timeout
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=3)
            except Exception:
                pass
            self._thread = None
        
        # Reset state
        self.playing = False
        self.paused = False
        self.current_url = None
        self.current_title = None
    
    def pause(self):
        """Pause playback"""
        if self.playing:
            self.paused = True
            
    def resume(self):
        """Resume playback"""
        if self.playing:
            self.paused = False
    
    def is_playing(self) -> bool:
        return self.playing
    
    def get_info(self) -> dict:
        """Get current stream info"""
        return {
            'playing': self.playing,
            'paused': self.paused,
            'title': self.current_title,
            'url': self.current_url
        }
    
    def set_volume(self, vol: float):
        """Set stream volume (0.0 - 2.0)"""
        self.volume = max(0.0, min(2.0, vol))

    def set_pitch(self, pitch: float):
        """Set pitch multiplier (1.0 = normal, 1.5 = chipmunk)"""
        self.pitch = pitch
    
    def set_trim(self, start: float, end: float):
        """Set trim times in seconds"""
        self.trim_start = max(0.0, start)
        self.trim_end = max(0.0, end)
    
    def get_duration(self, url: str) -> float:
        """Get duration of YouTube video in seconds"""
        # Check cache first
        cached_file, _ = self._get_cached_file(url)
        if cached_file:
            return self._get_file_duration(cached_file)
        
        # If not cached, extract info without downloading
        if not YTDLP_AVAILABLE:
            return 0.0
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return float(info.get('duration', 0))
        except Exception:
            return 0.0
    
    def _get_file_duration(self, filepath: str) -> float:
        """Get duration of audio file using ffprobe or pygame"""
        if not os.path.exists(filepath):
            return 0.0
        
        # Try ffprobe first
        try:
            if FFMPEG_PATH:
                ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
                ffprobe_exe = os.path.join(ffmpeg_dir, 'ffprobe.exe')
                if not os.path.exists(ffprobe_exe):
                    ffprobe_exe = os.path.join(ffmpeg_dir, 'ffprobe')
            else:
                ffprobe_exe = 'ffprobe'
            
            cmd = [
                ffprobe_exe,
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                filepath
            ]
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                timeout=5
            )
            
            if result.returncode == 0:
                output = result.stdout.decode().strip()
                if output:
                    return float(output)
        except Exception:
            pass
        
        # Fallback to pygame if available
        try:
            import pygame
            if pygame.mixer.get_init():
                sound = pygame.mixer.Sound(filepath)
                return sound.get_length()
        except Exception:
            pass
        
        return 0.0
