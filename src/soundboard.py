"""
Core Soundboard Logic - Hỗ trợ dual output (speakers + virtual cable)
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
import wave
import threading
from pathlib import Path

try:
    import pyaudio
    import numpy as np
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class Soundboard:
    def __init__(self, sounds_dir="sounds"):
        if pygame:
            pygame.mixer.init()
        self.sounds_dir = Path(sounds_dir)
        self.sounds = {}
        self.volume = 0.7
        self.load_sounds()
        
        # Virtual audio routing
        self.virtual_output_device = None
        self.routing_enabled = False
        self.pyaudio_instance = None
        
        # Track active streams để có thể stop
        self._active_streams = []
        self._streams_lock = threading.Lock()
        self._stop_flag = threading.Event()
        
        if PYAUDIO_AVAILABLE:
            self.pyaudio_instance = pyaudio.PyAudio()
    
    def load_sounds(self):
        """Load all audio files from sounds directory"""
        if not self.sounds_dir.exists():
            self.sounds_dir.mkdir(parents=True)
            return
        
        # Hỗ trợ nhiều định dạng
        extensions = ['*.wav', '*.mp3', '*.ogg', '*.flac']
        for ext in extensions:
            for file in self.sounds_dir.glob(ext):
                name = file.stem
                self.sounds[name] = str(file)
    
    def set_virtual_output(self, device_index):
        """Set virtual output device for routing to Discord/Games"""
        self.virtual_output_device = device_index
        self.routing_enabled = device_index is not None
    
    def play_sound(self, sound_name):
        """Play a sound - output to both speakers and virtual device"""
        if sound_name not in self.sounds:
            return False
        
        sound_path = self.sounds[sound_name]
        
        # Clear stop flag
        self._stop_flag.clear()
        
        # Play qua pygame (speakers)
        if pygame:
            try:
                sound = pygame.mixer.Sound(sound_path)
                sound.set_volume(self.volume)
                sound.play()
            except Exception as e:
                print(f"Pygame error: {e}")
        
        # Đồng thời route qua virtual device nếu enabled
        if self.routing_enabled and self.virtual_output_device is not None:
            threading.Thread(
                target=self._play_to_virtual_device,
                args=(sound_path,),
                daemon=True
            ).start()
        
        return True
    
    def _play_to_virtual_device(self, sound_path):
        """Play audio to virtual device (VB-Cable) in background thread"""
        if not PYAUDIO_AVAILABLE or self.pyaudio_instance is None:
            return
        
        try:
            # Đọc file WAV
            if sound_path.lower().endswith('.wav'):
                self._play_wav_to_device(sound_path)
            else:
                # Với các format khác, dùng pygame để convert
                self._play_other_format_to_device(sound_path)
        except Exception as e:
            print(f"Virtual device error: {e}")
    
    def _play_wav_to_device(self, wav_path):
        """Play WAV file directly to virtual device"""
        try:
            wf = wave.open(wav_path, 'rb')
            
            # Lấy sample rate của device (VB-Cable thường dùng 48000)
            device_info = self.pyaudio_instance.get_device_info_by_index(self.virtual_output_device)
            device_sample_rate = int(device_info['defaultSampleRate'])
            file_sample_rate = wf.getframerate()
            
            print(f"Playing WAV to virtual device: {wav_path}")
            print(f"  File rate: {file_sample_rate}, Device rate: {device_sample_rate}")
            print(f"  Channels: {wf.getnchannels()}, Sample width: {wf.getsampwidth()}")
            
            # Dùng sample rate của device
            stream = self.pyaudio_instance.open(
                format=self.pyaudio_instance.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=device_sample_rate,
                output=True,
                output_device_index=self.virtual_output_device
            )
            
            # Đọc toàn bộ file để resample nếu cần
            all_data = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(all_data, dtype=np.int16)
            
            # Reshape theo channels
            if wf.getnchannels() == 2:
                audio_array = audio_array.reshape(-1, 2)
            
            # Resample nếu sample rate khác
            if file_sample_rate != device_sample_rate:
                audio_array = self._resample_audio(audio_array, file_sample_rate, device_sample_rate)
            
            # Track stream
            with self._streams_lock:
                self._active_streams.append(stream)
            
            try:
                # Apply volume
                audio_array = (audio_array * self.volume).astype(np.int16)
                
                # Play in chunks để có thể stop
                chunk_size = 4096
                audio_bytes = audio_array.tobytes()
                
                for i in range(0, len(audio_bytes), chunk_size):
                    if self._stop_flag.is_set():
                        break
                    try:
                        chunk = audio_bytes[i:i + chunk_size]
                        stream.write(chunk)
                    except Exception:
                        break
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
                try:
                    wf.close()
                except Exception:
                    pass
                with self._streams_lock:
                    if stream in self._active_streams:
                        self._active_streams.remove(stream)
        except Exception as e:
            print(f"WAV playback error: {e}")
    
    def _play_other_format_to_device(self, sound_path):
        """Play non-WAV formats by loading with pygame and streaming"""
        try:
            # Load sound với pygame để decode
            sound = pygame.mixer.Sound(sound_path)
            
            # Lấy raw audio data từ pygame Sound
            try:
                raw_data = pygame.sndarray.array(sound)
            except Exception:
                print(f"Cannot convert {sound_path} to array, skipping virtual output")
                return
            
            # Lấy thông tin từ pygame mixer
            mixer_freq, mixer_size, mixer_channels = pygame.mixer.get_init()
            
            # Convert to int16 nếu cần
            if raw_data.dtype != np.int16:
                if raw_data.dtype == np.float32 or raw_data.dtype == np.float64:
                    raw_data = (raw_data * 32767).astype(np.int16)
                else:
                    raw_data = raw_data.astype(np.int16)
            
            # Determine channels từ shape
            if len(raw_data.shape) > 1:
                channels = raw_data.shape[1]
            else:
                channels = 1
                raw_data = raw_data.reshape(-1, 1)
            
            # Lấy sample rate của device
            device_info = self.pyaudio_instance.get_device_info_by_index(self.virtual_output_device)
            device_sample_rate = int(device_info['defaultSampleRate'])
            
            # Resample nếu sample rate khác nhau
            if mixer_freq != device_sample_rate:
                raw_data = self._resample_audio(raw_data, mixer_freq, device_sample_rate)
            
            stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=device_sample_rate,
                output=True,
                output_device_index=self.virtual_output_device
            )
            
            # Track stream
            with self._streams_lock:
                self._active_streams.append(stream)
            
            try:
                # Apply volume and play in chunks để có thể stop
                audio_data = (raw_data * self.volume).astype(np.int16)
                chunk_size = 4096
                audio_bytes = audio_data.tobytes()
                
                for i in range(0, len(audio_bytes), chunk_size):
                    if self._stop_flag.is_set():
                        break
                    try:
                        chunk = audio_bytes[i:i + chunk_size]
                        stream.write(chunk)
                    except Exception:
                        break
            finally:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
                with self._streams_lock:
                    if stream in self._active_streams:
                        self._active_streams.remove(stream)
        except Exception as e:
            print(f"Format conversion error: {e}")
    
    def _resample_audio(self, audio_data, src_rate, dst_rate):
        """Resample audio từ src_rate sang dst_rate"""
        if src_rate == dst_rate:
            return audio_data
        
        # Tính số samples mới
        duration = len(audio_data) / src_rate
        new_length = int(duration * dst_rate)
        
        # Resample bằng linear interpolation
        if len(audio_data.shape) > 1:
            # Stereo
            resampled = np.zeros((new_length, audio_data.shape[1]), dtype=np.int16)
            for ch in range(audio_data.shape[1]):
                resampled[:, ch] = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data[:, ch]
                ).astype(np.int16)
        else:
            # Mono
            resampled = np.interp(
                np.linspace(0, len(audio_data), new_length),
                np.arange(len(audio_data)),
                audio_data
            ).astype(np.int16)
        
        return resampled
    
    def stop_all(self):
        """Stop all playing sounds - cả speakers và virtual device"""
        # Stop pygame sounds
        if pygame:
            pygame.mixer.stop()
        
        # Set flag để các thread tự dừng (không close stream ở đây)
        self._stop_flag.set()
    
    def set_volume(self, volume):
        """Set global volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if pygame:
            # Không dùng music.set_volume vì ta dùng Sound objects
            pass
    
    def get_sound_list(self):
        """Get list of available sounds"""
        return sorted(list(self.sounds.keys()))
    
    def add_sound(self, file_path, name=None):
        """Add a new sound to the soundboard"""
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
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
