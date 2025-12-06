"""Sound Player - Core audio playback functionality"""
import threading
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


class SoundPlayer:
    """Handles sound file playback to speakers and VB-Cable"""
    
    def __init__(self, sounds_dir: str, vb_manager):
        self.sounds_dir = Path(sounds_dir)
        self.vb_manager = vb_manager
        self.sounds: dict[str, str] = {}
        self.volume = 0.7
        self.pitch = 1.0
        
        self._stop_flag = threading.Event()
        self._thread_id = 0
        self._vb_lock = threading.Lock()
        self._is_playing = False
        self._current_sound = None
        
        self._init_pygame()
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
        """Get list of sound names"""
        return sorted(self.sounds.keys())
    
    def set_volume(self, vol: float):
        """Set volume (0.0 - 50.0 for scream mode)"""
        self.volume = max(0.0, min(50.0, vol))
    
    def set_pitch(self, pitch: float):
        """Set pitch (0.5 - 2.0)"""
        self.pitch = max(0.5, min(2.0, pitch))
    
    def play(self, name: str) -> bool:
        """Play sound by name"""
        if name not in self.sounds:
            return False
        
        path = self.sounds[name]
        self._stop_flag.clear()
        self._current_sound = name
        
        # Play via pygame (speakers)
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
                snd = pygame.mixer.Sound(path)
                snd.set_volume(min(self.volume, 1.0))
                snd.play()
            except Exception as e:
                print(f"Audio error: {e}")
                return False
        
        # Route to VB-Cable
        if self.vb_manager.is_connected() and SD_AVAILABLE:
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
                snd = pygame.mixer.Sound(path)
                arr = pygame.sndarray.array(snd)
                freq = pygame.mixer.get_init()[0]
                vb_samplerate = self.vb_manager.get_samplerate()
                
                # Convert to float32
                if arr.dtype == np.int16:
                    audio = arr.astype(np.float32) / 32768.0
                elif arr.dtype == np.int32:
                    audio = arr.astype(np.float32) / 2147483648.0
                else:
                    audio = arr.astype(np.float32)
                
                # Resample if needed
                need_resample = (self.pitch != 1.0) or (freq != vb_samplerate)
                
                if need_resample:
                    pitch_factor = 1.0 / self.pitch if self.pitch != 1.0 else 1.0
                    rate_factor = vb_samplerate / freq if freq != vb_samplerate else 1.0
                    final_length = int(len(audio) * pitch_factor * rate_factor)
                    
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
                
                # Play to VB-Cable
                with sd.OutputStream(
                    device=self.vb_manager.device_id,
                    samplerate=vb_samplerate,
                    channels=audio.shape[1] if audio.ndim > 1 else 1,
                    dtype='float32',
                    blocksize=2048,
                ) as stream:
                    chunk_size = 2048
                    for i in range(0, len(audio), chunk_size):
                        if tid != self._thread_id or self._stop_flag.is_set():
                            break
                        
                        chunk = audio[i:i+chunk_size]
                        if chunk.ndim == 1:
                            chunk = chunk.reshape(-1, 1)
                        
                        stream.write(chunk)
                        
            except Exception as e:
                print(f"VB-Cable playback error: {e}")
            finally:
                if tid == self._thread_id:
                    self._is_playing = False
                    self._current_sound = None
    
    def stop(self):
        """Stop all sounds"""
        self._stop_flag.set()
        self._thread_id += 1
        self._current_sound = None
        
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        
        threading.Timer(0.2, self._stop_flag.clear).start()
    
    def is_playing(self) -> bool:
        return self._is_playing
    
    def get_current_sound(self) -> str:
        return self._current_sound
    
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
        """Cleanup resources"""
        self.stop()
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
            except Exception:
                pass
