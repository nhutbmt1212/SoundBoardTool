import threading
import time
import logging
from pathlib import Path
from ..logging.debug_logger import log_loop_action
from .sound_library import SoundLibrary
from .audio_utils import AudioUtils

# Configure logging
logger = logging.getLogger(__name__)

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
    
    # Constants
    STREAM_CHUNK_SIZE = 4096
    DEFAULT_VOLUME = 0.7
    MAX_VOLUME = 50.0
    MIN_PITCH = 0.5
    MAX_PITCH = 2.0
    PAUSE_CHECK_INTERVAL = 0.1
    
    def __init__(self, sounds_dir: str, vb_manager):
        self.library = SoundLibrary(sounds_dir)
        self.vb_manager = vb_manager
        
        # Playback configuration
        self.volume = self.DEFAULT_VOLUME
        self.pitch = 1.0
        self.trim_start = 0.0
        self.trim_end = 0.0
        self.loop_enabled = False
        
        # State
        self._stop_flag = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._thread_id = 0
        self._vb_lock = threading.Lock()
        self._is_playing = False
        self._is_paused = False
        self._current_sound = None
        
        # Audio effects
        from .effects_processor import AudioEffectsProcessor
        self.effects_processor = AudioEffectsProcessor()
        self.effects_config = {}
        
        self._init_pygame()
    
    @property
    def sounds(self) -> dict:
        """Facade for library sounds, kept for compatibility"""
        return self.library.sounds

    def _init_pygame(self):
        """Initialize pygame mixer"""
        if pygame:
            try:
                # Increased buffer from 512 to 2048 for better stability
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            except Exception as e:
                logger.debug(f"Failed to initialize pygame mixer with custom settings: {e}")
                try:
                    pygame.mixer.init()
                except Exception as e2:
                    logger.error(f"Failed to initialize pygame mixer: {e2}")
    
    # --- Delegation to SoundLibrary ---
    
    def load_sounds(self):
        self.library.load_sounds()
    
    def get_sounds(self) -> list[str]:
        return self.library.get_sounds()
        
    def add_sound(self, filepath: str, name: str = None) -> bool:
        return self.library.add_sound(filepath, name)
        
    def add_sound_from_data(self, filename: str, data: bytes) -> bool:
        return self.library.add_sound_from_data(filename, data)
    
    def delete_sound(self, name: str) -> bool:
        return self.library.delete_sound(name)

    # --- Delegation to AudioUtils ---

    def get_audio_duration(self, name: str) -> float:
        path = self.library.get_path(name)
        if not path:
            return 0.0
        return AudioUtils.get_audio_duration(path)
    
    def get_waveform_data(self, name: str, samples: int = 200) -> list:
        path = self.library.get_path(name)
        if not path:
            return []
        return AudioUtils.get_waveform_data(path, samples)

    # --- Playback Logic ---
    
    def set_volume(self, vol: float):
        """Set volume (0.0 - 50.0 for scream mode)"""
        self.volume = max(0.0, min(self.MAX_VOLUME, vol))
    
    def set_pitch(self, pitch: float):
        """Set pitch (0.5 - 2.0)"""
        self.pitch = max(self.MIN_PITCH, min(self.MAX_PITCH, pitch))
    
    def set_trim(self, start: float, end: float):
        """Set trim times in seconds"""
        self.trim_start = max(0.0, start)
        self.trim_end = max(0.0, end)
    
    def set_loop(self, enabled: bool):
        """Enable or disable loop infinity"""
        log_loop_action("SoundPlayer.set_loop", f"Enabled={enabled}")
        self.loop_enabled = enabled
    
    def get_loop(self) -> bool:
        """Get current loop state"""
        return self.loop_enabled
    
    def play(self, name: str) -> bool:
        """Play sound by name"""
        path = self.library.get_path(name)
        if not path:
            return False
        
        self._stop_flag.clear()
        self._pause_event.set()
        self._current_sound = name
        self._is_paused = False
        
        log_loop_action("SoundPlayer.play", f"Name={name}, LoopEnabled={self.loop_enabled}")

        # Play to both speaker and VB-Cable using sounddevice
        if SD_AVAILABLE and pygame and pygame.mixer.get_init():
            self._thread_id += 1
            
            # Play to speaker in background
            threading.Thread(
                target=self._play_speaker,
                args=(path, self._thread_id, name),
                daemon=True
            ).start()
            
            # Route to VB-Cable if connected
            if self.vb_manager.is_connected():
                threading.Thread(
                    target=self._play_vb,
                    args=(path, self._thread_id, name),
                    daemon=True
                ).start()
        elif pygame and pygame.mixer.get_init():
            # Fallback to pure pygame
            try:
                pygame.mixer.stop()
                snd = pygame.mixer.Sound(path)
                snd.set_volume(min(self.volume, 1.0))
                snd.play()
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
                return False
        
        return True

    def play_file(self, file_path: str, on_complete: callable = None) -> bool:
        """Play audio from file path directly (for TTS and temp files)
        
        Args:
            file_path: Absolute path to audio file
            on_complete: Optional callback called when playback ends (for cleanup)
            
        Returns:
            bool: True if playback started successfully
        """
        if not file_path or not Path(file_path).exists():
            return False
        
        self._stop_flag.clear()
        self._pause_event.set()
        self._current_sound = 'TTS'
        self._is_paused = False
        
        if SD_AVAILABLE and pygame and pygame.mixer.get_init():
            self._thread_id += 1
            
            # Play to speaker in background
            threading.Thread(
                target=self._play_file_internal,
                args=(file_path, self._thread_id, on_complete),
                daemon=True
            ).start()
            
            # Route to VB-Cable if connected
            if self.vb_manager.is_connected():
                threading.Thread(
                    target=self._play_file_vb,
                    args=(file_path, self._thread_id),
                    daemon=True
                ).start()
        elif pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
                snd = pygame.mixer.Sound(file_path)
                snd.set_volume(min(self.volume, 1.0))
                snd.play()
                if on_complete:
                    threading.Thread(target=self._wait_and_callback, args=(on_complete,), daemon=True).start()
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
                return False
        
        return True

    def _play_file_internal(self, file_path: str, tid: int, on_complete: callable = None):
        """Internal method to play file to speaker"""
        if not SD_AVAILABLE or not pygame or not pygame.mixer.get_init():
            return
        
        if tid != self._thread_id:
            return
        
        audio, freq = AudioUtils.load_and_convert_audio(file_path)
        if audio is None:
            if on_complete:
                on_complete()
            return
        
        processed_audio = AudioUtils.process_audio(
            audio, freq,
            trim_start=0,
            trim_end=0,
            pitch=self.pitch,
            target_samplerate=freq
        )
        
        self._stream_audio_to_device(processed_audio, freq, None, tid)
        
        # Call on_complete callback when done
        if on_complete:
            on_complete()

    def _play_file_vb(self, file_path: str, tid: int):
        """Play file to VB-Cable"""
        if not SD_AVAILABLE or not pygame or not pygame.mixer.get_init():
            return
        
        with self._vb_lock:
            if tid != self._thread_id:
                return
            
            self._is_playing = True
            try:
                audio, freq = AudioUtils.load_and_convert_audio(file_path)
                if audio is None:
                    return
                
                vb_samplerate = self.vb_manager.get_samplerate()
                
                processed_audio = AudioUtils.process_audio(
                    audio, freq,
                    trim_start=0,
                    trim_end=0,
                    pitch=self.pitch,
                    target_samplerate=vb_samplerate
                )
                
                self._stream_audio_to_device(processed_audio, vb_samplerate, self.vb_manager.device_id, tid)
            finally:
                if tid == self._thread_id:
                    self._is_playing = False
                    self._current_sound = None

    def _wait_and_callback(self, on_complete: callable):
        """Wait for pygame playback to end and call callback"""
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        if on_complete:
            on_complete()

    def pause(self):
        """Pause current playback"""
        if self._is_playing and not self._is_paused:
            self._is_paused = True
            self._pause_event.clear()
            return True
        return False
        
    def resume(self):
        """Resume current playback"""
        if self._is_playing and self._is_paused:
            self._is_paused = False
            self._pause_event.set()
            return True
        return False

    def is_paused(self) -> bool:
        return self._is_paused
        
    def is_playing(self) -> bool:
        return self._is_playing
    
    def get_current_sound(self) -> str:
        return self._current_sound

    def stop(self):
        """Stop all sounds"""
        self._stop_flag.set()
        self._pause_event.set()
        self._thread_id += 1
        self._current_sound = None
        self._is_playing = False
        self._is_paused = False
        
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
            except Exception as e:
                logger.debug(f"Failed to stop pygame mixer: {e}")
        
        threading.Timer(0.2, self._stop_flag.clear).start()

    def set_effects(self, effects_config: dict):
        self.effects_config = effects_config
    
    def get_effects(self) -> dict:
        return self.effects_config
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
            except Exception as e:
                logger.debug(f"Failed to quit pygame mixer: {e}")

    # --- Internal Streaming Logic ---

    def _stream_audio_to_device(self, audio, samplerate, device_id, tid):
        """Stream audio data to the specified output device"""
        try:
            channels = audio.shape[1] if audio.ndim > 1 else 1
            
            stream_kwargs = {
                'samplerate': samplerate,
                'channels': channels,
                'dtype': 'float32',
                'blocksize': self.STREAM_CHUNK_SIZE,
            }
            
            if device_id is not None:
                stream_kwargs['device'] = device_id
                
            # Use a slightly larger latency hint for better stability on Windows
            stream_kwargs['latency'] = 'high' if device_id is not None else 'low'
            
            with sd.OutputStream(**stream_kwargs) as stream:
                for i in range(0, len(audio), self.STREAM_CHUNK_SIZE):
                    if self._stop_flag.is_set() or tid != self._thread_id:
                        stream.abort()
                        return

                    # Handle Pause
                    if not self._pause_event.is_set():
                        while not self._pause_event.is_set():
                             if self._stop_flag.is_set() or tid != self._thread_id:
                                stream.abort()
                                return
                             time.sleep(0.1)
                    
                    chunk = audio[i:i + self.STREAM_CHUNK_SIZE]
                    
                    # Apply volume
                    chunk = chunk * self.volume

                    # Continuous soft clipping to prevent distortion without chunk-boundary pops
                    # Uses a hybrid approach: linear up to 0.8, then smooth arc to 1.0
                    abs_chunk = np.abs(chunk)
                    mask = abs_chunk > 0.8
                    if np.any(mask):
                        # Simple soft clipper that is transparent below 0.8
                        # y = 0.8 + 0.2 * tanh((x - 0.8) / 0.2)
                        # This ensures a smooth transition and continuous derivative
                        chunk = np.where(
                            mask,
                            np.sign(chunk) * (0.8 + 0.19 * np.tanh((abs_chunk - 0.8) / 0.19)),
                            chunk
                        )
                    
                    # Apply effects
                    if self.effects_config:
                        chunk = self.effects_processor.apply_effects(chunk, self.effects_config)
                    
                    if chunk.ndim == 1:
                        chunk = chunk.reshape(-1, 1)
                    stream.write(chunk)
        except Exception as e:
            msg = str(e)
            if "AUDCLNT_E_DEVICE_INVALIDATED" not in msg and "There is no driver installed" not in msg:
                 logger.error(f"Error streaming audio: {e}")

    def _play_speaker(self, path: str, tid: int, name: str):
        """Play to speaker in background thread"""
        if not SD_AVAILABLE or not pygame or not pygame.mixer.get_init():
            return
        
        if tid != self._thread_id:
            return
            
        audio, freq = AudioUtils.load_and_convert_audio(path)
        if audio is None:
            return

        while tid == self._thread_id and not self._stop_flag.is_set():
            log_loop_action("SoundPlayer._play_speaker", f"Loop Start | Local Playback | Name={name} | LoopEnabled={self.loop_enabled}")
            
            # Process audio
            processed_audio = AudioUtils.process_audio(
                audio, freq, 
                trim_start=self.trim_start, 
                trim_end=self.trim_end, 
                pitch=self.pitch, 
                target_samplerate=freq
            )
            
            self._stream_audio_to_device(processed_audio, freq, None, tid)
            
            if not self.loop_enabled or self._stop_flag.is_set() or tid != self._thread_id:
                log_loop_action("SoundPlayer._play_speaker", f"Loop Exit | Name={name} | LoopEnabled={self.loop_enabled} | StopFlag={self._stop_flag.is_set()}")
                break
            
            log_loop_action("SoundPlayer._play_speaker", f"Loop Restarting | Name={name}")
            time.sleep(0.005)

    def _play_vb(self, path: str, tid: int, sound_name: str):
        """Play to VB-Cable in background thread"""
        if not SD_AVAILABLE or not pygame or not pygame.mixer.get_init():
            return
        
        with self._vb_lock:
            if tid != self._thread_id:
                return
            
            self._is_playing = True
            try:
                while tid == self._thread_id and not self._stop_flag.is_set():
                    log_loop_action("SoundPlayer._play_vb", f"Loop Start | VB Playback | Name={sound_name} | LoopEnabled={self.loop_enabled}")
                    
                    audio, freq = AudioUtils.load_and_convert_audio(path)
                    if audio is None:
                        break

                    vb_samplerate = self.vb_manager.get_samplerate()
                    
                    processed_audio = AudioUtils.process_audio(
                        audio, freq,
                        trim_start=self.trim_start,
                        trim_end=self.trim_end,
                        pitch=self.pitch,
                        target_samplerate=vb_samplerate
                    )

                    self._stream_audio_to_device(processed_audio, vb_samplerate, self.vb_manager.device_id, tid)
                    
                    if not self.loop_enabled or self._stop_flag.is_set() or tid != self._thread_id:
                        log_loop_action("SoundPlayer._play_vb", f"Loop Exit | Name={sound_name} | LoopEnabled={self.loop_enabled} | StopFlag={self._stop_flag.is_set()}")
                        break
                    
                    log_loop_action("SoundPlayer._play_vb", f"Loop Restarting | Name={sound_name}")
                    time.sleep(0.005)
                        
            finally:
                if tid == self._thread_id:
                    self._is_playing = False
                    self._current_sound = None
