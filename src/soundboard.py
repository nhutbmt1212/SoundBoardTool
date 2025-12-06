"""
Core Soundboard Logic - Sử dụng sounddevice cho audio routing
"""
try:
    import pygame
    AUDIO_BACKEND = 'pygame'
except ImportError:
    try:
        import pygame_ce as pygame
        AUDIO_BACKEND = 'pygame_ce'
    except ImportError:
        pygame = None
        AUDIO_BACKEND = None

import os
import threading
import traceback
from pathlib import Path

try:
    import sounddevice as sd
    import numpy as np
    from scipy.io import wavfile
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None
    np = None


def log_debug(msg):
    """Debug logging"""
    print(f"[DEBUG] {msg}")


class Soundboard:
    def __init__(self, sounds_dir="sounds"):
        if pygame:
            pygame.mixer.init()
        self.sounds_dir = Path(sounds_dir)
        self.sounds = {}
        self.volume = 0.7
        self.load_sounds()
        
        # Virtual audio routing
        self.virtual_device_id = None
        self.routing_enabled = False
        self._stop_flag = threading.Event()
        self._is_playing_vb = False  # Track nếu đang play qua VB-Cable
        self._vb_lock = threading.Lock()  # Lock để tránh race condition
        self._current_thread_id = 0  # ID của thread hiện tại
        
        # Auto-detect VB-Cable
        if SOUNDDEVICE_AVAILABLE:
            self._auto_detect_vb_cable()
    
    def _auto_detect_vb_cable(self):
        """Tự động tìm và kết nối VB-Cable"""
        try:
            devices = sd.query_devices()
            
            # Tìm VB-Cable output device
            for i, dev in enumerate(devices):
                if dev['max_output_channels'] > 0:
                    name = dev['name'].lower()
                    if 'vb-audio virtual cable' in name:
                        self.virtual_device_id = i
                        self.routing_enabled = True
                        print(f"✅ VB-Cable found: {dev['name']} (id={i})")
                        return True
            
            print("⚠️ VB-Cable not found")
            return False
        except Exception as e:
            print(f"Error detecting VB-Cable: {e}")
            return False
    
    def is_vb_cable_connected(self):
        """Kiểm tra VB-Cable đã kết nối chưa"""
        return self.routing_enabled and self.virtual_device_id is not None
    
    def load_sounds(self):
        """Load all audio files from sounds directory"""
        if not self.sounds_dir.exists():
            self.sounds_dir.mkdir(parents=True)
            return
        
        extensions = ['*.wav', '*.mp3', '*.ogg', '*.flac']
        for ext in extensions:
            for file in self.sounds_dir.glob(ext):
                name = file.stem
                self.sounds[name] = str(file)
    
    def play_sound(self, sound_name):
        """Play sound - speakers + VB-Cable"""
        if sound_name not in self.sounds:
            return False
        
        sound_path = self.sounds[sound_name]
        self._stop_flag.clear()
        
        # Stop sound cũ trước khi play mới
        if pygame:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        
        # Play qua speakers (pygame)
        if pygame:
            try:
                sound = pygame.mixer.Sound(sound_path)
                sound.set_volume(self.volume)
                sound.play()
            except Exception as e:
                print(f"Pygame error: {e}")
        
        # Route qua VB-Cable
        if self.routing_enabled and self.virtual_device_id is not None:
            # Tăng thread ID để cancel thread cũ
            self._current_thread_id += 1
            thread_id = self._current_thread_id
            
            threading.Thread(
                target=self._play_to_vb_cable,
                args=(sound_path, thread_id),
                daemon=True
            ).start()
        
        return True
    
    def _play_to_vb_cable(self, sound_path, thread_id):
        """Play audio to VB-Cable"""
        if not SOUNDDEVICE_AVAILABLE:
            return
        
        # Acquire lock để chỉ 1 thread play tại một thời điểm
        with self._vb_lock:
            # Kiểm tra xem thread này còn là thread mới nhất không
            if thread_id != self._current_thread_id:
                return
            
            self._is_playing_vb = True
            
            try:
                # Load audio với pygame và convert
                sound = pygame.mixer.Sound(sound_path)
                audio_array = pygame.sndarray.array(sound)
                
                # Get mixer settings
                mixer_freq, _, _ = pygame.mixer.get_init()
                
                # Convert to float32 cho sounddevice
                if audio_array.dtype == np.int16:
                    audio_float = audio_array.astype(np.float32) / 32768.0
                elif audio_array.dtype == np.int32:
                    audio_float = audio_array.astype(np.float32) / 2147483648.0
                else:
                    audio_float = audio_array.astype(np.float32)
                
                # Apply volume
                audio_float = audio_float * self.volume
                
                # Kiểm tra stop flag và thread ID trước khi play
                if self._stop_flag.is_set() or thread_id != self._current_thread_id:
                    self._is_playing_vb = False
                    return
                
                # Stop any existing stream trước khi play mới
                try:
                    sd.stop()
                except Exception:
                    pass
                
                # Play to VB-Cable
                sd.play(audio_float, samplerate=mixer_freq, device=self.virtual_device_id)
                
                # Wait với check stop flag định kỳ
                while True:
                    try:
                        # Kiểm tra thread ID - nếu có thread mới hơn thì dừng
                        if thread_id != self._current_thread_id:
                            try:
                                sd.stop()
                            except Exception:
                                pass
                            break
                        
                        stream = sd.get_stream()
                        if stream is None or not stream.active:
                            break
                        
                        if self._stop_flag.is_set():
                            try:
                                sd.stop()
                            except Exception:
                                pass
                            break
                        
                        sd.sleep(50)
                    except Exception:
                        break
                
            except Exception:
                pass
            finally:
                # Chỉ reset flag nếu đây là thread cuối cùng
                if thread_id == self._current_thread_id:
                    self._is_playing_vb = False
    
    def stop_all(self):
        """Stop all sounds"""
        # Set flag để thread dừng
        self._stop_flag.set()
        
        # Tăng thread ID để invalidate tất cả thread đang chạy
        self._current_thread_id += 1
        
        # Stop pygame
        if pygame:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        
        # Reset flag sau một chút để cho phép play tiếp
        def reset_flag():
            try:
                self._stop_flag.clear()
            except Exception:
                pass
        
        threading.Timer(0.2, reset_flag).start()
    
    def set_volume(self, volume):
        """Set volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
    
    def get_sound_list(self):
        """Get list of available sounds"""
        return sorted(list(self.sounds.keys()))
    
    def add_sound(self, file_path, name=None):
        """Add a new sound"""
        path = Path(file_path)
        if not path.exists():
            return False
        
        sound_name = name or path.stem
        dest = self.sounds_dir / f"{sound_name}{path.suffix}"
        
        if path != dest:
            import shutil
            shutil.copy(path, dest)
        
        self.sounds[sound_name] = str(dest)
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_all()
