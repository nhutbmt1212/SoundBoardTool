"""Audio Engine - Handles sound playback and VB-Cable routing"""
import threading
import subprocess
import os
from pathlib import Path

try:
    import pygame
except ImportError:
    try:
        import pygame_ce as pygame
    except ImportError:
        pygame = None

try:
    import sounddevice as sd
    import numpy as np
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    sd = None
    np = None

# YouTube streaming
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    yt_dlp = None


def find_ffmpeg():
    """Find ffmpeg executable"""
    import shutil
    
    # Check if in PATH
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    
    # Common Windows locations
    common_paths = [
        os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\WinGet\Packages'),
        r'C:\ffmpeg\bin',
        r'C:\Program Files\ffmpeg\bin',
        os.path.expandvars(r'%USERPROFILE%\ffmpeg\bin'),
    ]
    
    for base in common_paths:
        if os.path.exists(base):
            # Search for ffmpeg.exe
            for root, dirs, files in os.walk(base):
                if 'ffmpeg.exe' in files:
                    return os.path.join(root, 'ffmpeg.exe')
    
    return None


FFMPEG_PATH = find_ffmpeg()


class AudioEngine:
    def __init__(self, sounds_dir: str = "sounds"):
        self.sounds_dir = Path(sounds_dir)
        self.sounds: dict[str, str] = {}
        self.volume = 0.7
        self.pitch = 1.0  # 1.0 = normal, 1.5 = chipmunk, 2.0 = super high
        
        # VB-Cable
        self._vb_device_id = None
        self._vb_enabled = False
        self._vb_samplerate = None  # Cache sample rate
        
        # Mic passthrough
        self._mic_device_id = None
        self._mic_input_stream = None
        self._mic_output_stream = None
        self._mic_enabled = False
        self._mic_volume = 1.0
        self._mic_buffer = None
        
        # YouTube streaming
        self._yt_process = None
        self._yt_thread = None
        self._yt_stop_event = threading.Event()
        self._yt_playing = False
        self._yt_volume = 1.0
        self._yt_current_url = None
        self._yt_current_title = None
        
        # Thread control
        self._stop_flag = threading.Event()
        self._thread_id = 0
        self._vb_lock = threading.Lock()
        self._is_playing = False
        self._current_playing_sound = None
        self._sound_finished_callback = None
        
        # Init
        self._init_pygame()
        self._detect_vb_cable()
        self._detect_mic()
        self.load_sounds()
    
    def _init_pygame(self):
        """Initialize pygame mixer"""
        if pygame:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            except Exception:
                try:
                    pygame.mixer.init()
                except Exception:
                    pass
    
    def _detect_vb_cable(self):
        """Auto-detect VB-Cable device - prefer 2-channel MME/DirectSound for compatibility"""
        if not SD_AVAILABLE:
            return
        try:
            vb_devices = []
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_output_channels'] > 0:
                    name_lower = dev['name'].lower()
                    if 'vb-audio virtual cable' in name_lower and 'speakers' in name_lower:
                        vb_devices.append((i, dev))
            
            if not vb_devices:
                return
            
            # Prefer device with exactly 2 channels (MME/DirectSound) - more stable
            for i, dev in vb_devices:
                if dev['max_output_channels'] == 2:
                    self._vb_device_id = i
                    self._vb_enabled = True
                    print(f"✓ VB-Cable: [{i}] {dev['name']} ({dev['max_output_channels']} ch)")
                    return
            
            # Fallback to first device
            i, dev = vb_devices[0]
            self._vb_device_id = i
            self._vb_enabled = True
            print(f"✓ VB-Cable: [{i}] {dev['name']} ({dev['max_output_channels']} ch)")
            
            # Cache sample rate
            if self._vb_device_id is not None:
                try:
                    vb_info = sd.query_devices(self._vb_device_id)
                    self._vb_samplerate = int(vb_info['default_samplerate'])
                except Exception:
                    self._vb_samplerate = 48000  # Default
        except Exception as e:
            print(f"VB-Cable detection error: {e}")
    
    def _detect_mic(self):
        """Auto-detect default microphone"""
        if not SD_AVAILABLE:
            return
        try:
            # Get default input device
            default_input = sd.default.device[0]
            if default_input is not None and default_input >= 0:
                dev = sd.query_devices(default_input)
                if dev['max_input_channels'] > 0:
                    self._mic_device_id = default_input
                    print(f"✓ Mic: {dev['name']}")
        except Exception:
            pass
    
    def get_mic_devices(self) -> list:
        """Get list of available microphones"""
        if not SD_AVAILABLE:
            return []
        devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    devices.append({'id': i, 'name': dev['name']})
        except Exception:
            pass
        return devices
    
    def set_mic_device(self, device_id: int):
        """Set microphone device"""
        was_enabled = self._mic_enabled
        if was_enabled:
            self.stop_mic_passthrough()
        self._mic_device_id = device_id
        if was_enabled:
            self.start_mic_passthrough()
    
    def get_current_mic_id(self) -> int:
        return self._mic_device_id
    
    def set_mic_volume(self, vol: float):
        """Set mic passthrough volume"""
        self._mic_volume = max(0.0, min(2.0, vol))
    
    def start_mic_passthrough(self):
        """Start routing mic to VB-Cable using separate streams"""
        if not SD_AVAILABLE or self._mic_device_id is None or self._vb_device_id is None:
            print(f"Cannot start mic: SD={SD_AVAILABLE}, mic={self._mic_device_id}, vb={self._vb_device_id}")
            return False
        
        if self._mic_input_stream is not None:
            return True  # Already running
        
        try:
            import queue
            
            # Shared buffer between input and output
            self._mic_buffer = queue.Queue(maxsize=20)
            engine = self
            
            # Get sample rates
            vb_info = sd.query_devices(self._vb_device_id)
            vb_samplerate = int(vb_info['default_samplerate'])
            
            mic_info = sd.query_devices(self._mic_device_id)
            mic_samplerate = int(mic_info['default_samplerate'])
            
            # Use common sample rate (prefer VB-Cable's rate)
            samplerate = vb_samplerate
            blocksize = 512
            
            print(f"Mic passthrough: mic={mic_samplerate}Hz, vb={vb_samplerate}Hz, using={samplerate}Hz")
            
            # Store for resampling
            self._mic_resample_ratio = mic_samplerate / vb_samplerate if mic_samplerate != vb_samplerate else None
            
            def input_callback(indata, frames, time, status):
                """Capture mic input and resample if needed"""
                if status:
                    print(f"Mic input: {status}")
                try:
                    data = indata.copy() * engine._mic_volume
                    
                    # Resample if needed
                    if engine._mic_resample_ratio and engine._mic_resample_ratio != 1.0:
                        from scipy import signal
                        target_length = int(len(data) / engine._mic_resample_ratio)
                        data = signal.resample(data, target_length)
                    
                    engine._mic_buffer.put_nowait(data)
                except queue.Full:
                    pass
                except Exception as e:
                    print(f"Mic resample error: {e}")
            
            def output_callback(outdata, frames, time, status):
                """Output to VB-Cable"""
                if status:
                    print(f"VB output: {status}")
                try:
                    data = engine._mic_buffer.get_nowait()
                    # Handle size mismatch
                    if len(data) >= len(outdata):
                        outdata[:] = data[:len(outdata)]
                    else:
                        outdata[:len(data)] = data
                        outdata[len(data):] = 0
                except queue.Empty:
                    outdata[:] = 0
            
            # Create input stream (from mic) - use mic's native rate
            self._mic_input_stream = sd.InputStream(
                device=self._mic_device_id,
                samplerate=mic_samplerate,
                channels=1,
                dtype='float32',
                callback=input_callback,
                blocksize=blocksize,
            )
            
            # Create output stream (to VB-Cable) - use VB-Cable's rate
            self._mic_output_stream = sd.OutputStream(
                device=self._vb_device_id,
                samplerate=vb_samplerate,
                channels=1,
                dtype='float32',
                callback=output_callback,
                blocksize=blocksize,
                latency='low'
            )
            
            # Start both streams
            self._mic_input_stream.start()
            self._mic_output_stream.start()
            self._mic_enabled = True
            
            if mic_samplerate != vb_samplerate:
                print(f"✓ Mic passthrough started with resampling ({mic_samplerate}Hz → {vb_samplerate}Hz)")
            else:
                print(f"✓ Mic passthrough started ({samplerate}Hz)")
            
            return True
            
        except Exception as e:
            print(f"Mic passthrough error: {e}")
            import traceback
            traceback.print_exc()
            self.stop_mic_passthrough()
            return False
    
    def stop_mic_passthrough(self):
        """Stop mic passthrough"""
        if self._mic_input_stream is not None:
            try:
                self._mic_input_stream.stop()
                self._mic_input_stream.close()
            except Exception:
                pass
            self._mic_input_stream = None
        
        if self._mic_output_stream is not None:
            try:
                self._mic_output_stream.stop()
                self._mic_output_stream.close()
            except Exception:
                pass
            self._mic_output_stream = None
        
        self._mic_buffer = None
        self._mic_enabled = False
    
    def is_mic_enabled(self) -> bool:
        return self._mic_enabled
    
    def is_vb_connected(self) -> bool:
        return self._vb_enabled and self._vb_device_id is not None
    
    # === YouTube Streaming ===
    
    def play_youtube(self, url: str) -> dict:
        """Stream YouTube audio directly to VB-Cable"""
        if not YTDLP_AVAILABLE:
            return {'success': False, 'error': 'yt-dlp not installed'}
        
        if not SD_AVAILABLE or self._vb_device_id is None:
            return {'success': False, 'error': 'VB-Cable not available'}
        
        # Stop any existing YouTube stream
        self.stop_youtube()
        
        try:
            # Get audio URL and info
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
                audio_url = info.get('url')
                
                if not audio_url:
                    # Find audio format
                    for fmt in info.get('formats', []):
                        if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                            audio_url = fmt.get('url')
                            break
                    if not audio_url:
                        # Fallback to any format with audio
                        for fmt in info.get('formats', []):
                            if fmt.get('acodec') != 'none':
                                audio_url = fmt.get('url')
                                break
                
                if not audio_url:
                    return {'success': False, 'error': 'No audio stream found'}
                
                self._yt_current_url = url
                self._yt_current_title = title
                self._yt_stop_event.clear()
                
                # Start streaming thread
                self._yt_thread = threading.Thread(
                    target=self._youtube_stream_loop,
                    args=(audio_url,),
                    daemon=True
                )
                self._yt_thread.start()
                self._yt_playing = True
                
                print(f"▶ YouTube: {title}")
                return {'success': True, 'title': title}
                
        except Exception as e:
            print(f"YouTube error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _youtube_stream_loop(self, audio_url: str):
        """Background thread to stream YouTube audio via ffmpeg"""
        try:
            # Find ffmpeg
            ffmpeg_exe = FFMPEG_PATH or 'ffmpeg'
            print(f"Using ffmpeg: {ffmpeg_exe}")
            
            # Use ffmpeg to decode audio stream
            ffmpeg_cmd = [
                ffmpeg_exe,
                '-reconnect', '1',
                '-reconnect_streamed', '1', 
                '-reconnect_delay_max', '5',
                '-i', audio_url,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-'
            ]
            
            # Start ffmpeg process
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            self._yt_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=8192,
                startupinfo=startupinfo
            )
            
            # Check if process started
            import time
            time.sleep(0.5)
            if self._yt_process.poll() is not None:
                # Process ended immediately - get error
                stderr = self._yt_process.stderr.read().decode('utf-8', errors='ignore')
                print(f"FFmpeg error: {stderr[:500]}")
                return
            
            print("FFmpeg started, detecting VB-Cable sample rate...")
            
            # Get VB-Cable device's supported sample rate
            vb_info = sd.query_devices(self._vb_device_id)
            samplerate = int(vb_info['default_samplerate'])
            channels = min(2, vb_info['max_output_channels'])
            blocksize = 8192  # Larger buffer for stability
            bytes_per_sample = 2 * channels  # 16-bit stereo
            
            print(f"VB-Cable requires: rate={samplerate}Hz, ch={channels}")
            
            # Need to restart ffmpeg with correct sample rate
            print("Restarting ffmpeg with correct sample rate...")
            self._yt_process.terminate()
            self._yt_process.wait(timeout=2)
            
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
            
            self._yt_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=8192,
                startupinfo=startupinfo
            )
            
            import time
            time.sleep(0.3)
            print(f"FFmpeg restarted with rate={samplerate}Hz")
            
            print(f"Opening streams: VB-Cable device={self._vb_device_id}, rate={samplerate}, ch={channels}")
            
            # Open VB-Cable stream with larger buffer to prevent underruns
            vb_stream = sd.OutputStream(
                device=self._vb_device_id,
                samplerate=samplerate,
                channels=channels,
                dtype='float32',
                blocksize=blocksize,
                latency='high',  # Use high latency for stability
                clip_off=True
            )
            vb_stream.start()
            print(f"✓ VB-Cable stream opened")
            
            # Also play to default speaker - use same sample rate to avoid distortion
            speaker_stream = None
            try:
                default_output = sd.default.device[1]
                if default_output is not None and default_output != self._vb_device_id:
                    # Use VB-Cable's sample rate for speaker too
                    speaker_stream = sd.OutputStream(
                        device=default_output,
                        samplerate=samplerate,  # Same rate as VB-Cable
                        channels=channels,
                        dtype='float32',
                        blocksize=blocksize,
                        latency='high',
                        clip_off=True
                    )
                    speaker_stream.start()
                    print(f"✓ Speaker stream opened (rate={samplerate}Hz)")
            except Exception as e:
                print(f"Speaker output not available: {e}")
            
            try:
                frame_count = 0
                while not self._yt_stop_event.is_set():
                    # Read raw PCM data from ffmpeg
                    raw_data = self._yt_process.stdout.read(blocksize * bytes_per_sample)
                    if not raw_data:
                        # Check if process ended with error
                        if self._yt_process.poll() is not None:
                            stderr = self._yt_process.stderr.read().decode('utf-8', errors='ignore')
                            if stderr:
                                print(f"FFmpeg ended: {stderr[:300]}")
                        break
                    
                    # Convert to float32
                    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    audio = audio.reshape(-1, channels)
                    
                    # Apply volume
                    audio *= self._yt_volume
                    audio_out = np.ascontiguousarray(audio)
                    
                    # Write to VB-Cable (for Discord)
                    vb_stream.write(audio_out)
                    
                    # Write to speaker (for monitoring)
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
            if not self._yt_stop_event.is_set():
                print(f"YouTube stream error: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._cleanup_youtube_process()
            self._yt_playing = False
            print("YouTube stream ended")
    
    def _cleanup_youtube_process(self):
        """Clean up ffmpeg process"""
        if self._yt_process:
            try:
                self._yt_process.terminate()
                self._yt_process.wait(timeout=2)
            except Exception:
                try:
                    self._yt_process.kill()
                except Exception:
                    pass
            self._yt_process = None
    
    def stop_youtube(self):
        """Stop YouTube streaming"""
        self._yt_stop_event.set()
        self._cleanup_youtube_process()
        
        if self._yt_thread:
            try:
                self._yt_thread.join(timeout=2)
            except Exception:
                pass
            self._yt_thread = None
        
        self._yt_playing = False
        self._yt_current_url = None
        self._yt_current_title = None
    
    def is_youtube_playing(self) -> bool:
        return self._yt_playing
    
    def get_youtube_info(self) -> dict:
        """Get current YouTube stream info"""
        return {
            'playing': self._yt_playing,
            'title': self._yt_current_title,
            'url': self._yt_current_url
        }
    
    def set_youtube_volume(self, vol: float):
        """Set YouTube stream volume (0.0 - 2.0)"""
        self._yt_volume = max(0.0, min(2.0, vol))
    
    def load_sounds(self):
        """Load sounds from directory"""
        self.sounds.clear()
        if not self.sounds_dir.exists():
            self.sounds_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for ext in ('*.wav', '*.mp3', '*.ogg', '*.flac'):
            for f in self.sounds_dir.glob(ext):
                self.sounds[f.stem] = str(f)
    
    def get_sounds(self) -> list[str]:
        return sorted(self.sounds.keys())
    
    def set_volume(self, vol: float):
        # Allow up to 50.0 for scream mode (5000% boost)
        self.volume = max(0.0, min(50.0, vol))
    
    def set_pitch(self, pitch: float):
        # 1.0 = normal, 1.5 = chipmunk, 2.0 = super high
        self.pitch = max(0.5, min(2.0, pitch))
    
    def play(self, name: str) -> bool:
        """Play sound by name"""
        if name not in self.sounds:
            return False
        
        path = self.sounds[name]
        self._stop_flag.clear()
        self._current_playing_sound = name
        
        # Stop previous & play via pygame
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
                snd = pygame.mixer.Sound(path)
                snd.set_volume(min(self.volume, 1.0))
                snd.play()
            except Exception as e:
                print(f"Audio error: {e}")
                return False
        
        # Route to VB-Cable (with pitch support)
        if self._vb_enabled and SD_AVAILABLE:
            self._thread_id += 1
            threading.Thread(
                target=self._play_vb,
                args=(path, self._thread_id, name),
                daemon=True
            ).start()
        
        return True
    
    def _play_vb(self, path: str, tid: int, sound_name: str):
        """Play to VB-Cable in background thread"""
        if not SD_AVAILABLE or not pygame or not pygame.mixer.get_init():
            return
        
        with self._vb_lock:
            if tid != self._thread_id:
                return
            
            self._is_playing = True
            try:
                # Load & convert audio
                snd = pygame.mixer.Sound(path)
                arr = pygame.sndarray.array(snd)
                freq = pygame.mixer.get_init()[0]
                
                # Use cached sample rate
                vb_samplerate = self._vb_samplerate or 48000
                
                # Convert to float32 first
                if arr.dtype == np.int16:
                    audio = arr.astype(np.float32) / 32768.0
                elif arr.dtype == np.int32:
                    audio = arr.astype(np.float32) / 2147483648.0
                else:
                    audio = arr.astype(np.float32)
                
                # Fast resample using numpy (linear interpolation)
                need_resample = (self.pitch != 1.0) or (freq != vb_samplerate)
                
                if need_resample:
                    # Calculate final length
                    pitch_factor = 1.0 / self.pitch if self.pitch != 1.0 else 1.0
                    rate_factor = vb_samplerate / freq if freq != vb_samplerate else 1.0
                    final_length = int(len(audio) * pitch_factor * rate_factor)
                    
                    # Fast linear interpolation
                    if audio.ndim == 1:
                        indices = np.linspace(0, len(audio) - 1, final_length)
                        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
                    else:
                        indices = np.linspace(0, len(audio) - 1, final_length)
                        resampled = []
                        for ch in range(audio.shape[1]):
                            resampled.append(np.interp(indices, np.arange(len(audio)), audio[:, ch]))
                        audio = np.column_stack(resampled).astype(np.float32)
                
                audio *= self.volume
                
                if self._stop_flag.is_set() or tid != self._thread_id:
                    return
                
                # Use OutputStream instead of sd.play() to avoid device conflicts
                try:
                    with sd.OutputStream(
                        device=self._vb_device_id,
                        samplerate=vb_samplerate,
                        channels=audio.shape[1] if audio.ndim > 1 else 1,
                        dtype='float32',
                        blocksize=2048,
                    ) as stream:
                        print(f"✓ Sound stream opened")
                        
                        # Write audio in chunks
                        chunk_size = 2048
                        for i in range(0, len(audio), chunk_size):
                            if tid != self._thread_id or self._stop_flag.is_set():
                                break
                            
                            chunk = audio[i:i+chunk_size]
                            
                            # Ensure chunk is 2D
                            if chunk.ndim == 1:
                                chunk = chunk.reshape(-1, 1)
                            
                            stream.write(chunk)
                        
                        print(f"✓ Sound playback complete")
                        
                except Exception as e:
                    print(f"✗ Error playing to VB-Cable: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception:
                pass
            finally:
                if tid == self._thread_id:
                    self._is_playing = False
                    self._current_playing_sound = None
    
    def stop(self):
        """Stop all sounds"""
        self._stop_flag.set()
        self._thread_id += 1
        
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        
        threading.Timer(0.2, self._stop_flag.clear).start()
    
    def add_sound(self, filepath: str, name: str = None) -> bool:
        """Add sound file to library"""
        import shutil
        src = Path(filepath)
        if not src.exists():
            return False
        
        dest_name = name or src.stem
        dest = self.sounds_dir / f"{dest_name}{src.suffix}"
        
        try:
            if src != dest:
                shutil.copy(src, dest)
            self.sounds[dest_name] = str(dest)
            return True
        except Exception:
            return False
    
    def delete_sound(self, name: str) -> bool:
        """Delete sound from library"""
        if name not in self.sounds:
            return False
        try:
            Path(self.sounds[name]).unlink(missing_ok=True)
            del self.sounds[name]
            return True
        except Exception:
            return False
    
    def cleanup(self):
        self.stop()
        self.stop_mic_passthrough()
        self.stop_youtube()
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
            except Exception:
                pass
