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
        
        # Mic passthrough
        self._mic_device_id = None
        self._mic_input_stream = None
        self._mic_output_stream = None
        self._mic_enabled = False
        self._mic_volume = 1.0
        self._mic_buffer = None
        
        # Thread control
        self._stop_flag = threading.Event()
        self._thread_id = 0
        self._vb_lock = threading.Lock()
        self._is_playing = False
        
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
            mic_info = sd.query_devices(self._mic_device_id)
            samplerate = int(mic_info['default_samplerate'])
            blocksize = 512
            
            def input_callback(indata, frames, time, status):
                """Capture mic input"""
                if status:
                    print(f"Input: {status}")
                try:
                    # Apply volume and put in buffer
                    data = indata.copy() * engine._mic_volume
                    engine._mic_buffer.put_nowait(data)
                except queue.Full:
                    pass  # Drop frame if buffer full
            
            def output_callback(outdata, frames, time, status):
                """Output to VB-Cable"""
                if status:
                    print(f"Output: {status}")
                try:
                    data = engine._mic_buffer.get_nowait()
                    outdata[:len(data)] = data
                    if len(data) < len(outdata):
                        outdata[len(data):] = 0
                except queue.Empty:
                    outdata[:] = 0  # Silence if no data
            
            # Create input stream (from mic)
            self._mic_input_stream = sd.InputStream(
                device=self._mic_device_id,
                samplerate=samplerate,
                channels=1,
                dtype='float32',
                callback=input_callback,
                blocksize=blocksize,
                latency='low'
            )
            
            # Create output stream (to VB-Cable)
            self._mic_output_stream = sd.OutputStream(
                device=self._vb_device_id,
                samplerate=samplerate,
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
            print(f"✓ Mic passthrough started (rate={samplerate})")
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
        self.stop_mic_passthrough()
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
            except Exception:
                pass
