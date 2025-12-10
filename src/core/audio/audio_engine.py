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
    
    # Constants
    DEBOUNCE_SECONDS = 0.5  # Minimum time between playback commands
    
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

    def pause_sound(self):
        """Pause sound playback"""
        self.sound_player.pause()
        
    def resume_sound(self):
        """Resume sound playback"""
        self.sound_player.resume()
    
    def set_sound_loop(self, enabled: bool):
        """Enable/disable loop for sound playback"""
        self.sound_player.set_loop(enabled)
    
    def get_sound_loop(self) -> bool:
        """Get sound loop state"""
        return self.sound_player.get_loop()
    
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
            if current_time - self._last_play_time < self.DEBOUNCE_SECONDS:
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
    def _is_paused(self) -> bool:
        return self.sound_player.is_paused()
    
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
    
    # === YouTube ===
    
    def _play_stream(self, stream_obj, url: str, progress_callback=None, loop: bool = False) -> dict:
        """Generic stream playback with download and exclusive lock
        
        Args:
            stream_obj: Stream object (youtube or tiktok)
            url: Video URL
            progress_callback: Optional progress callback
            loop: Loop enabled flag
            
        Returns:
            dict: {'success': bool, 'error': str (optional), 'title': str (optional)}
        """
        # Step 1: Download/Cache OUTSIDE the lock to allow concurrency
        filepath, title = stream_obj.download(url, progress_callback)
        if not filepath:
            return {'success': False, 'error': 'Failed to download video'}
        
        # Step 2: Playback INSIDE the lock (exclusive access)
        with self._playback_lock:
            current_time = time.time()
            self._last_play_time = current_time
            
            # Stop everything else
            self._stop_all_internal()
            
            # Play stream (instant as it hits cache)
            return stream_obj.play(url, None, loop)
    
    def play_youtube(self, url: str, progress_callback=None, loop: bool = False) -> dict:
        """Play YouTube video - ensures exclusive playback"""
        return self._play_stream(self.youtube, url, progress_callback, loop)
    
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
    
    def set_youtube_loop(self, enabled: bool):
        """Enable/disable loop for YouTube playback"""
        self.youtube.set_loop(enabled)
    
    def get_youtube_loop(self) -> bool:
        """Get YouTube loop state"""
        return self.youtube.get_loop()
    
    # === TikTok ===
    
    def play_tiktok(self, url: str, progress_callback=None, loop: bool = False) -> dict:
        """Play TikTok video - ensures exclusive playback"""
        return self._play_stream(self.tiktok, url, progress_callback, loop)
    
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
    
    def set_tiktok_loop(self, enabled: bool):
        """Enable/disable loop for TikTok playback"""
        self.tiktok.set_loop(enabled)
    
    def get_tiktok_loop(self) -> bool:
        """Get TikTok loop state"""
        return self.tiktok.get_loop()
    
    # === Audio Effects ===
    
    def set_sound_effects(self, effects_config: dict):
        """Set effects for sound playback"""
        self.sound_player.set_effects(effects_config)
    
    def get_sound_effects(self) -> dict:
        """Get current sound effects"""
        return self.sound_player.get_effects()
    
    def set_youtube_effects(self, effects_config: dict):
        """Set effects for YouTube playback"""
        self.youtube.set_effects(effects_config)
    
    def get_youtube_effects(self) -> dict:
        """Get current YouTube effects"""
        return self.youtube.get_effects()
    
    def set_tiktok_effects(self, effects_config: dict):
        """Set effects for TikTok playback"""
        self.tiktok.set_effects(effects_config)
    
    def get_tiktok_effects(self) -> dict:
        """Get current TikTok effects"""
        return self.tiktok.get_effects()
    
    # === Cleanup ===
    
    def cleanup(self):
        """Cleanup all resources"""
        with self._playback_lock:
            self._stop_all_internal()
            self.mic.stop()

