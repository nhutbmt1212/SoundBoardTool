"""API Layer - YouTube Endpoints"""
import eel
from .base_stream_api import BaseStreamAPI


class YouTubeAPI(BaseStreamAPI):
    """YouTube-related API endpoints"""
    
    def __init__(self, audio_engine, sounds_dir):
        super().__init__(audio_engine, sounds_dir, stream_type='youtube')
    
    def _register_endpoints(self):
        """Register all YouTube endpoints with Eel"""
        eel.expose(self.play_youtube)
        eel.expose(self.stop_youtube)
        eel.expose(self.pause_youtube)
        eel.expose(self.resume_youtube)
        eel.expose(self.get_youtube_info)
        eel.expose(self.is_youtube_playing)
        eel.expose(self.set_youtube_volume)
        eel.expose(self.get_youtube_items)
        eel.expose(self.add_youtube_item)
        eel.expose(self.delete_youtube_item)
        eel.expose(self.save_youtube_as_sound)
        eel.expose(self.get_youtube_duration)
    
    def _get_stream_object(self):
        """Get YouTube stream object from audio engine"""
        return self.audio.youtube
    
    def _get_keybinds_key(self) -> str:
        """Get settings key for YouTube keybinds"""
        return 'youtubeKeybinds'
    
    # Public API methods - delegate to base class
    
    def play_youtube(self, url: str, volume: float = 1.0, pitch: float = 1.0, 
                     start_time: float = 0, end_time: float = 0):
        """Play YouTube audio by URL with specific settings"""
        return self._play(url, volume, pitch, start_time, end_time)
    
    def stop_youtube(self):
        """Stop YouTube streaming"""
        self._stop()
    
    def pause_youtube(self):
        """Pause YouTube streaming"""
        self._pause()
    
    def resume_youtube(self):
        """Resume YouTube streaming"""
        self._resume()
    
    def get_youtube_info(self):
        """Get YouTube playback info"""
        return self._get_info()
    
    def is_youtube_playing(self):
        """Check if YouTube is playing"""
        return self._is_playing()
    
    def set_youtube_volume(self, vol: float):
        """Set YouTube volume"""
        return self._set_volume(vol)
    
    def get_youtube_items(self):
        """Get all YouTube cached items"""
        return self._get_items()
    
    def add_youtube_item(self, url: str):
        """Add YouTube item (download and cache)"""
        return self._add_item(url)
    
    def delete_youtube_item(self, url: str):
        """Delete YouTube cached item"""
        return self._delete_item(url)
    
    def save_youtube_as_sound(self, url: str):
        """Save YouTube cache as a sound item"""
        return self._save_as_sound(url)
    
    def get_youtube_duration(self, url: str):
        """Get YouTube video duration in seconds"""
        return self._get_duration(url)
