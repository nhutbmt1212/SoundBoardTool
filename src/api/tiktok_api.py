"""API Layer - TikTok Endpoints"""
import eel
from .base_stream_api import BaseStreamAPI


class TikTokAPI(BaseStreamAPI):
    """TikTok-related API endpoints"""
    
    def __init__(self, audio_engine, sounds_dir):
        super().__init__(audio_engine, sounds_dir, stream_type='tiktok')
    
    def _register_endpoints(self):
        """Register all TikTok endpoints with Eel"""
        eel.expose(self.play_tiktok)
        eel.expose(self.stop_tiktok)
        eel.expose(self.pause_tiktok)
        eel.expose(self.resume_tiktok)
        eel.expose(self.get_tiktok_info)
        eel.expose(self.is_tiktok_playing)
        eel.expose(self.set_tiktok_volume)
        eel.expose(self.get_tiktok_items)
        eel.expose(self.add_tiktok_item)
        eel.expose(self.delete_tiktok_item)
        eel.expose(self.save_tiktok_as_sound)
        eel.expose(self.get_tiktok_duration)
    
    def _get_stream_object(self):
        """Get TikTok stream object from audio engine"""
        return self.audio.tiktok
    
    def _get_keybinds_key(self) -> str:
        """Get settings key for TikTok keybinds"""
        return 'tiktokKeybinds'
    
    # Public API methods - delegate to base class
    
    def play_tiktok(self, url: str, volume: float = 1.0, pitch: float = 1.0, 
                    start_time: float = 0, end_time: float = 0):
        """Play TikTok audio by URL with specific settings"""
        return self._play(url, volume, pitch, start_time, end_time)
    
    def stop_tiktok(self):
        """Stop TikTok streaming"""
        self._stop()
    
    def pause_tiktok(self):
        """Pause TikTok streaming"""
        self._pause()
    
    def resume_tiktok(self):
        """Resume TikTok streaming"""
        self._resume()
    
    def get_tiktok_info(self):
        """Get TikTok playback info"""
        return self._get_info()
    
    def is_tiktok_playing(self):
        """Check if TikTok is playing"""
        return self._is_playing()
    
    def set_tiktok_volume(self, vol: float):
        """Set TikTok volume"""
        return self._set_volume(vol)
    
    def get_tiktok_items(self):
        """Get all TikTok cached items"""
        return self._get_items()
    
    def add_tiktok_item(self, url: str):
        """Add TikTok item (download and cache)"""
        return self._add_item(url)
    
    def delete_tiktok_item(self, url: str):
        """Delete TikTok cached item"""
        return self._delete_item(url)
    
    def save_tiktok_as_sound(self, url: str):
        """Save TikTok cache as a sound item"""
        return self._save_as_sound(url)
    
    def get_tiktok_duration(self, url: str):
        """Get TikTok video duration in seconds"""
        return self._get_duration(url)
