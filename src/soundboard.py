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
            threading.Thread(
                target=self._play_to_vb_cable,
                args=(sound_path,),
                daemon=True
            ).start()
        
        return True
    
    def _play_to_vb_cable(self, sound_path):
        """Play audio to VB-Cable"""
        log_debug("_play_to_vb_cable: START")
        
        if not SOUNDDEVICE_AVAILABLE:
            log_debug("_play_to_vb_cable: sounddevice not available")
            return
        
        self._is_playing_vb = True
        log_debug("_play_to_vb_cable: set _is_playing_vb = True")
        
        try:
            log_debug("_play_to_vb_cable: loading sound with pygame")
            # Load audio với pygame và convert
            sound = pygame.mixer.Sound(sound_path)
            audio_array = pygame.sndarray.array(sound)
            log_debug(f"_play_to_vb_cable: audio_array shape={audio_array.shape}, dtype={audio_array.dtype}")
            
            # Get mixer settings
            mixer_freq, _, _ = pygame.mixer.get_init()
            log_debug(f"_play_to_vb_cable: mixer_freq={mixer_freq}")
            
            # Convert to float32 cho sounddevice
            if audio_array.dtype == np.int16:
                audio_float = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_float = audio_array.astype(np.float32) / 2147483648.0
            else:
                audio_float = audio_array.astype(np.float32)
            
            # Apply volume
            audio_float = audio_float * self.volume
            log_debug(f"_play_to_vb_cable: audio_float ready, shape={audio_float.shape}")
            
            # Kiểm tra stop flag trước khi play
            if self._stop_flag.is_set():
                log_debug("_play_to_vb_cable: stop flag set before play, returning")
                self._is_playing_vb = False
                return
            
            log_debug(f"_play_to_vb_cable: calling sd.play() device={self.virtual_device_id}")
            # Play to VB-Cable
            sd.play(audio_float, samplerate=mixer_freq, device=self.virtual_device_id)
            log_debug("_play_to_vb_cable: sd.play() called, entering wait loop")
            
            # Wait với check stop flag định kỳ thay vì block hoàn toàn
            loop_count = 0
            while True:
                loop_count += 1
                try:
                    stream = sd.get_stream()
                    if stream is None or not stream.active:
                        log_debug(f"_play_to_vb_cable: stream ended (loop={loop_count})")
                        break
                    
                    if self._stop_flag.is_set():
                        log_debug(f"_play_to_vb_cable: stop flag detected (loop={loop_count})")
                        try:
                            sd.stop()
                            log_debug("_play_to_vb_cable: sd.stop() called successfully")
                        except Exception as stop_err:
                            log_debug(f"_play_to_vb_cable: sd.stop() error: {stop_err}")
                        break
                    
                    sd.sleep(50)  # Check mỗi 50ms
                except Exception as loop_err:
                    log_debug(f"_play_to_vb_cable: loop error: {loop_err}")
                    break
            
            log_debug("_play_to_vb_cable: wait loop finished")
            
        except Exception as e:
            log_debug(f"_play_to_vb_cable: EXCEPTION: {e}")
            log_debug(traceback.format_exc())
        finally:
            self._is_playing_vb = False
            log_debug("_play_to_vb_cable: END, _is_playing_vb = False")
    
    def stop_all(self):
        """Stop all sounds"""
        log_debug("stop_all: START")
        
        # Set flag trước để thread tự dừng (thread sẽ gọi sd.stop())
        log_debug("stop_all: setting stop flag")
        self._stop_flag.set()
        
        # Stop pygame
        log_debug("stop_all: stopping                   ")
        if pygame:
            try:
                pygame.mixer.stop()
                log_debug("stop_all: pygame.mixer.stop() OK")
            except Exception as e:
                log_debug(f"stop_all: pygame.mixer.stop() error: {e}")
        
        # KHÔNG gọi sd.stop() ở đây - để thread tự xử lý khi thấy stop flag
        # Tránh race condition khi cả 2 nơi cùng gọi sd.stop()
        log_debug(f"stop_all: sounddevice will be stopped by thread (is_playing={self._is_playing_vb})")
        
        # Reset flag sau một chút để cho phép play tiếp
        def reset_flag():
            try:
                log_debug("stop_all: resetting stop flag")
                self._stop_flag.clear()
            except Exception as e:
                log_debug(f"stop_all: reset flag error: {e}")
        
        threading.Timer(0.3, reset_flag).start()
        log_debug("stop_all: END")
    
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
