"""Audio Engine - Handles sound playback and VB-Cable routing"""
import threading
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


class AudioEngine:
    def __init__(self, sounds_dir: str = "sounds"):
        self.sounds_dir = Path(sounds_dir)
        self.sounds: dict[str, str] = {}
        self.volume = 0.7
        self.pitch = 1.0  # 1.0 = normal, 1.5 = chipmunk, 2.0 = super high
        
        # VB-Cable
        self._vb_device_id = None
        self._vb_enabled = False
        
        # Thread control
        self._stop_flag = threading.Event()
        self._thread_id = 0
        self._vb_lock = threading.Lock()
        self._is_playing = False
        
        # Init
        self._init_pygame()
        self._detect_vb_cable()
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
        """Auto-detect VB-Cable device"""
        if not SD_AVAILABLE:
            return
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_output_channels'] > 0:
                    if 'vb-audio virtual cable' in dev['name'].lower():
                        self._vb_device_id = i
                        self._vb_enabled = True
                        print(f"✓ VB-Cable: {dev['name']}")
                        return
        except Exception:
            pass
    
    def is_vb_connected(self) -> bool:
        return self._vb_enabled and self._vb_device_id is not None
    
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
                args=(path, self._thread_id),
                daemon=True
            ).start()
        
        return True
    
    def _play_vb(self, path: str, tid: int):
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
                
                # Apply pitch by changing sample rate
                # Higher pitch = higher sample rate playback
                play_freq = int(freq * self.pitch)
                
                # Convert to float32
                if arr.dtype == np.int16:
                    audio = arr.astype(np.float32) / 32768.0
                elif arr.dtype == np.int32:
                    audio = arr.astype(np.float32) / 2147483648.0
                else:
                    audio = arr.astype(np.float32)
                
                audio *= self.volume
                
                if self._stop_flag.is_set() or tid != self._thread_id:
                    return
                
                # Stop existing & play
                try:
                    sd.stop()
                except Exception:
                    pass
                
                sd.play(audio, samplerate=play_freq, device=self._vb_device_id)
                
                # Wait loop
                while True:
                    if tid != self._thread_id or self._stop_flag.is_set():
                        try:
                            sd.stop()
                        except Exception:
                            pass
                        break
                    
                    stream = sd.get_stream()
                    if not stream or not stream.active:
                        break
                    
                    sd.sleep(50)
            except Exception:
                pass
            finally:
                if tid == self._thread_id:
                    self._is_playing = False
    
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
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
            except Exception:
                pass
