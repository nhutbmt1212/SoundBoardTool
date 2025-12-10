"""Base Streaming - Shared logic for streaming audio to VB-Cable"""
import threading
import subprocess
import os
import json
import hashlib
import sys
import time
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Add project root to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.utils import find_ffmpeg, kill_process_tree

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


FFMPEG_PATH = find_ffmpeg()

# Constants
FFMPEG_STARTUP_DELAY = 0.5  # Seconds to wait for FFmpeg to start
STREAM_BLOCKSIZE = 8192     # Audio buffer block size
AUDIO_CLIP_MIN = -1.0       # Minimum audio sample value
AUDIO_CLIP_MAX = 1.0        # Maximum audio sample value
THREAD_JOIN_TIMEOUT = 3.0   # Seconds to wait for thread cleanup


class BaseStream:
    """Base class for audio streaming with persistent cache"""
    
    def __init__(self, vb_manager, cache_dir_path: str):
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
        
        # Cache directory
        self.cache_dir = Path(cache_dir_path)
        self.cache_index_file = self.cache_dir / 'index.json'
        self._cache_index = self._load_cache_index()
        
        # Audio effects
        from .effects_processor import AudioEffectsProcessor
        self.effects_processor = AudioEffectsProcessor()
        self.effects_config = {}
        
        # To be overridden by subclasses if needed
        self.ytdl_format = 'bestaudio/best'

    def _load_cache_index(self) -> dict:
        """Load cache index from file"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to file"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_key(self, url: str) -> str:
        """Create cache key from URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached_file(self, url: str) -> tuple:
        """Check if URL is cached. Return (filepath, title) or (None, None)"""
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
            # Try common audio extensions
            for ext in ['.m4a', '.webm', '.opus', '.mp3', '.mp4']:
                new_path = filepath.with_suffix(ext)
                shutil.move(str(filepath), str(new_path))
                return new_path
        except Exception as e:
            logger.debug(f"Failed to fix cached file extension: {e}")
        return None
    
    def _download_and_cache(self, url: str, progress_callback=None) -> tuple:
        """Download video to cache. Return (filepath, title)"""
        cache_key = self._get_cache_key(url)
        output_template = self.cache_dir / f"{cache_key}.%(ext)s"
        
        hooks = []
        if progress_callback:
            hooks.append(progress_callback)
            
        ydl_opts = {
            'format': self.ytdl_format,
            'outtmpl': str(output_template),
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': hooks,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                
                # Find downloaded file
                output_file = None
                for ext in ['.m4a', '.webm', '.opus', '.mp3', '.mp4', '']:
                    test_file = self.cache_dir / f"{cache_key}{ext}"
                    if test_file.exists():
                        output_file = test_file
                        break
                
                if not output_file:
                    return None, None
                
                # Save to index
                self._cache_index[cache_key] = {
                    'url': url,
                    'title': title,
                    'file': str(output_file)
                }
                self._save_cache_index()
                
                return str(output_file), title
                
        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            return None, None
    
    def download(self, url: str, progress_callback=None) -> tuple:
        """Download video to cache without playing. Return (filepath, title)"""
        cached_file, title = self._get_cached_file(url)
        if cached_file:
            return cached_file, title
        return self._download_and_cache(url, progress_callback)
    
    def play(self, url: str, progress_callback=None) -> dict:
        """Stream audio to VB-Cable (with persistent cache)"""
        if not YTDLP_AVAILABLE:
            return {'success': False, 'error': 'yt-dlp not installed'}
        
        if not SD_AVAILABLE or not self.vb_manager.is_connected():
            return {'success': False, 'error': 'VB-Cable not available'}
        
        self.stop()
        
        try:
            cached_file, title = self._get_cached_file(url)
            
            if cached_file:
                audio_source = cached_file
            else:
                cached_file, title = self._download_and_cache(url, progress_callback)
                if not cached_file:
                    return {'success': False, 'error': 'Failed to download'}
                audio_source = cached_file
            
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

    def _stream_loop(self, audio_source: str):
        """Background thread to stream audio via ffmpeg"""
        try:
            ffmpeg_exe = FFMPEG_PATH or 'ffmpeg'
            samplerate = self.vb_manager.get_samplerate()
            channels = self.vb_manager.get_channels()
            blocksize = STREAM_BLOCKSIZE
            bytes_per_sample = 2 * channels
            
            is_local_file = os.path.exists(audio_source)
            
            # Build FFmpeg command
            filter_complex = []
            if abs(self.pitch - 1.0) > 0.01:
                new_rate = int(samplerate * self.pitch)
                filter_complex.append(f"asetrate={new_rate}")
                
            cmd = [ffmpeg_exe]
            
            if not is_local_file:
                cmd.extend([
                    '-reconnect', '1',
                    '-reconnect_streamed', '1',
                    '-reconnect_delay_max', '5'
                ])
            
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
            
            time.sleep(FFMPEG_STARTUP_DELAY)
            
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
                while not self._stop_event.is_set():
                    if self.paused:
                        time.sleep(0.1)
                        continue

                    raw_data = self._process.stdout.read(blocksize * bytes_per_sample)
                    if not raw_data:
                        break
                    
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    audio = audio.reshape(-1, channels)
                    
                    # Apply volume and clipping if needed
                    audio *= self.volume
                    if self.volume > 1.0:
                         audio = np.clip(audio, AUDIO_CLIP_MIN, AUDIO_CLIP_MAX)
                    
                    # Apply audio effects
                    if self.effects_config:
                        audio = self.effects_processor.apply_effects(audio, self.effects_config)
                         
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
                logger.error(f"Stream loop error: {e}")
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
            logger.debug(f"Failed to open speaker stream: {e}")
        return None
    
    def _cleanup_process(self):
        """Clean up ffmpeg process"""
        if self._process:
            try:
                kill_process_tree(self._process.pid)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    def stop(self):
        """Stop streaming"""
        self._stop_event.set()
        self._cleanup_process()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=THREAD_JOIN_TIMEOUT)
            except Exception as e:
                logger.debug(f"Thread join timeout: {e}")
            self._thread = None
        self.playing = False
        self.paused = False
        self.current_url = None
        self.current_title = None
    
    def pause(self):
        if self.playing:
            self.paused = True
            
    def resume(self):
        if self.playing:
            self.paused = False
    
    def is_playing(self) -> bool:
        return self.playing
    
    def get_info(self) -> dict:
        return {
            'playing': self.playing,
            'paused': self.paused,
            'title': self.current_title,
            'url': self.current_url
        }
    
    def set_volume(self, vol: float):
        self.volume = max(0.0, min(50.0, vol)) # Allow high volume for scream mode

    def set_pitch(self, pitch: float):
        self.pitch = pitch
    
    def set_trim(self, start: float, end: float):
        self.trim_start = max(0.0, start)
        self.trim_end = max(0.0, end)
    
    def set_effects(self, effects_config: dict):
        """Set effects configuration
        
        Args:
            effects_config: Dictionary of effect settings
        """
        self.effects_config = effects_config
    
    def get_effects(self) -> dict:
        """Get current effects configuration
        
        Returns:
            Dictionary of effect settings
        """
        return self.effects_config
        
    def get_duration(self, url: str) -> float:
        """Get duration of media in seconds"""
        cached_file, _ = self._get_cached_file(url)
        if cached_file:
            return self._get_file_duration(cached_file)
        
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
        except Exception as e:
            logger.debug(f"Failed to get duration for {url}: {e}")
            return 0.0

    def _get_file_duration(self, filepath: str) -> float:
        """Get duration of audio file"""
        if not os.path.exists(filepath):
            return 0.0
        
        # Try ffprobe
        try:
            if FFMPEG_PATH:
                ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
                ffprobe_exe = os.path.join(ffmpeg_dir, 'ffprobe.exe')
                if not os.path.exists(ffprobe_exe):
                    ffprobe_exe = os.path.join(ffmpeg_dir, 'ffprobe')
            else:
                ffprobe_exe = 'ffprobe'
            
            cmd = [
                ffprobe_exe, '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', filepath
            ]
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo, timeout=5)
            if result.returncode == 0:
                output = result.stdout.decode().strip()
                if output:
                    return float(output)
        except Exception:
            pass
        
        # Fallback to pygame
        try:
            import pygame
            if pygame.mixer.get_init():
                sound = pygame.mixer.Sound(filepath)
                return sound.get_length()
        except Exception:
            pass
        
        return 0.0
