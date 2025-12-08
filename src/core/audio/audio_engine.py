"""Audio Engine - Facade class combining all audio components"""
import time
import threading
from .vb_cable_manager import VBCableManager
from .sound_player import SoundPlayer
from .mic_passthrough import MicPassthrough
from .youtube_stream import YouTubeStream
from .tiktok_stream import TikTokStream


class AudioEngine:
    """
    Main audio engine facade that combines:
    - VB-Cable management
    - Sound playback
    - Microphone passthrough
    - YouTube streaming
    - TikTok streaming
    
    Ensures only ONE audio source plays at a time using mutex lock.
    """
    
    def __init__(self, sounds_dir: str = "sounds"):
        # Initialize components
        self.vb_manager = VBCableManager()
        self.sound_player = SoundPlayer(sounds_dir, self.vb_manager)
        self.mic = MicPassthrough(self.vb_manager)
        self.youtube = YouTubeStream(self.vb_manager)
        self.tiktok = TikTokStream(self.vb_manager)
        
        # Playback control
        self._last_play_time = 0
        self._playback_lock = threading.Lock()  # Mutex to prevent concurrent playback
    
    # === Sound Playback ===
    
    @property
    def sounds(self) -> dict:
        return self.sound_player.sounds
    
    @property
    def volume(self) -> float:
        return self.sound_player.volume
    
    @volume.setter
    def volume(self, val: float):
        self.sound_player.volume = val
    
    @property
    def pitch(self) -> float:
        return self.sound_player.pitch
    
    @pitch.setter
    def pitch(self, val: float):
        self.sound_player.pitch = val
    
    def load_sounds(self):
        self.sound_player.load_sounds()
    
    def get_sounds(self) -> list[str]:
        return self.sound_player.get_sounds()
    
    def set_volume(self, vol: float):
        self.sound_player.set_volume(vol)
    
    def set_pitch(self, pitch: float):
        self.sound_player.set_pitch(pitch)
    
    def set_trim(self, start: float, end: float):
        """Set trim times for sound playback"""
        self.sound_player.set_trim(start, end)
    
    def _stop_all_internal(self):
        """Internal method to stop all audio without lock (called within locked context)"""
        self.sound_player.stop()
        self.youtube.stop()
        self.tiktok.stop()
    
    def play(self, name: str) -> bool:
        """Play a local sound file - ensures exclusive playback"""
        with self._playback_lock:
            # Debounce check
            current_time = time.time()
            if current_time - self._last_play_time < 0.5:
                return False
            
            self._last_play_time = current_time
            
            # Stop everything else
            self._stop_all_internal()
            
            # Play the sound
            return self.sound_player.play(name)
    
    def stop(self):
        """Stop all audio playback"""
        with self._playback_lock:
            self._stop_all_internal()
    
    def add_sound(self, filepath: str, name: str = None) -> bool:
        return self.sound_player.add_sound(filepath, name)
    
    def delete_sound(self, name: str) -> bool:
        return self.sound_player.delete_sound(name)
    
    def get_audio_duration(self, name: str) -> float:
        """Get audio file duration in seconds"""
        return self.sound_player.get_audio_duration(name)
    
    def get_waveform_data(self, name: str, samples: int = 200) -> list:
        """Get waveform data for visualization"""
        return self.sound_player.get_waveform_data(name, samples)
    
    @property
    def _is_playing(self) -> bool:
        return self.sound_player.is_playing()
    
    @property
    def _current_playing_sound(self) -> str:
        return self.sound_player.get_current_sound()
    
    # === VB-Cable ===
    
    def is_vb_connected(self) -> bool:
        return self.vb_manager.is_connected()
    
    @property
    def _vb_enabled(self) -> bool:
        return self.vb_manager.enabled
    
    @property
    def _vb_device_id(self):
        return self.vb_manager.device_id
    
    # === Microphone ===
    
    def get_mic_devices(self) -> list:
        return self.mic.get_devices()
    
    def set_mic_device(self, device_id: int):
        self.mic.set_device(device_id)
    
    def get_current_mic_id(self) -> int:
        return self.mic.device_id
    
    def set_mic_volume(self, vol: float):
        self.mic.set_volume(vol)
    
    def start_mic_passthrough(self) -> bool:
        return self.mic.start()
    
    def stop_mic_passthrough(self):
        self.mic.stop()
    
    def is_mic_enabled(self) -> bool:
        return self.mic.is_enabled()
    
    # === YouTube ===
    
    def play_youtube(self, url: str, progress_callback=None) -> dict:
        """Play YouTube video - ensures exclusive playback"""
        with self._playback_lock:
            # Debounce check
            current_time = time.time()
            if current_time - self._last_play_time < 0.5:
                return {'success': False, 'error': 'Too many requests'}
            
            self._last_play_time = current_time
            
            # Stop everything else
            self._stop_all_internal()
            
            # Play YouTube
            return self.youtube.play(url, progress_callback)
    
    def stop_youtube(self):
        self.youtube.stop()
        
    def pause_youtube(self):
        self.youtube.pause()
        
    def resume_youtube(self):
        self.youtube.resume()
    
    def is_youtube_playing(self) -> bool:
        return self.youtube.is_playing()
    
    def get_youtube_info(self) -> dict:
        return self.youtube.get_info()
    
    def set_youtube_volume(self, vol: float):
        self.youtube.set_volume(vol)
        
    def set_youtube_pitch(self, pitch: float):
        self.youtube.set_pitch(pitch)
    
    def set_youtube_trim(self, start: float, end: float):
        self.youtube.set_trim(start, end)
    
    # === TikTok ===
    
    def play_tiktok(self, url: str, progress_callback=None) -> dict:
        """Play TikTok video - ensures exclusive playback"""
        with self._playback_lock:
            # Debounce check
            current_time = time.time()
            if current_time - self._last_play_time < 0.5:
                return {'success': False, 'error': 'Too many requests'}
            
            self._last_play_time = current_time
            
            # Stop everything else
            self._stop_all_internal()
            
            # Play TikTok
            return self.tiktok.play(url, progress_callback)
    
    def stop_tiktok(self):
        self.tiktok.stop()
        
    def pause_tiktok(self):
        self.tiktok.pause()
        
    def resume_tiktok(self):
        self.tiktok.resume()
    
    def is_tiktok_playing(self) -> bool:
        return self.tiktok.is_playing()
    
    def get_tiktok_info(self) -> dict:
        return self.tiktok.get_info()
    
    def set_tiktok_volume(self, vol: float):
        self.tiktok.set_volume(vol)
        
    def set_tiktok_pitch(self, pitch: float):
        self.tiktok.set_pitch(pitch)
    
    def set_tiktok_trim(self, start: float, end: float):
        self.tiktok.set_trim(start, end)
    
    # === Cleanup ===
    
    def cleanup(self):
        """Cleanup all resources"""
        with self._playback_lock:
            self._stop_all_internal()
            self.mic.stop()

