"""Audio Engine - Facade class combining all audio components"""
from .vb_cable_manager import VBCableManager
from .sound_player import SoundPlayer
from .mic_passthrough import MicPassthrough
from .youtube_stream import YouTubeStream


class AudioEngine:
    """
    Main audio engine facade that combines:
    - VB-Cable management
    - Sound playback
    - Microphone passthrough
    - YouTube streaming
    """
    
    def __init__(self, sounds_dir: str = "sounds"):
        # Initialize components
        self.vb_manager = VBCableManager()
        self.sound_player = SoundPlayer(sounds_dir, self.vb_manager)
        self.mic = MicPassthrough(self.vb_manager)
        self.youtube = YouTubeStream(self.vb_manager)
    
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
    
    def play(self, name: str) -> bool:
        # Enforce priority: Stop YouTube if playing
        self.youtube.stop()
        return self.sound_player.play(name)
    
    def stop(self):
        self.sound_player.stop()
        self.youtube.stop()
    
    def add_sound(self, filepath: str, name: str = None) -> bool:
        return self.sound_player.add_sound(filepath, name)
    
    def delete_sound(self, name: str) -> bool:
        return self.sound_player.delete_sound(name)
    
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
    
    def play_youtube(self, url: str) -> dict:
        # Enforce priority: Stop Sound if playing
        self.sound_player.stop()
        return self.youtube.play(url)
    
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
    
    # === Cleanup ===
    
    def cleanup(self):
        """Cleanup all resources"""
        self.sound_player.cleanup()
        self.mic.stop()
        self.youtube.stop()
