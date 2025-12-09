"""Base API Layer for Streaming Services (YouTube/TikTok)"""
import eel
import re
from abc import ABC, abstractmethod


class BaseStreamAPI(ABC):
    """Base class for streaming service APIs (YouTube, TikTok)
    
    Eliminates code duplication by providing common functionality.
    Subclasses must implement abstract methods to specify service-specific behavior.
    """
    
    # Pre-compiled regex for ANSI color code removal (performance optimization)
    ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*m')
    
    def __init__(self, audio_engine, sounds_dir, stream_type: str):
        """Initialize base stream API
        
        Args:
            audio_engine: AudioEngine instance
            sounds_dir: Directory for sound files
            stream_type: Type of stream ('youtube' or 'tiktok')
        """
        self.audio = audio_engine
        self.sounds_dir = sounds_dir
        self.stream_type = stream_type
        self._register_endpoints()
    
    @abstractmethod
    def _register_endpoints(self):
        """Register Eel endpoints - must be implemented by subclass"""
        pass
    
    @abstractmethod
    def _get_stream_object(self):
        """Get the stream object from audio engine - must be implemented by subclass"""
        pass
    
    @abstractmethod
    def _get_keybinds_key(self) -> str:
        """Get the settings key for keybinds - must be implemented by subclass"""
        pass
    
    def _play(self, url: str, volume: float = 1.0, pitch: float = 1.0, 
              start_time: float = 0, end_time: float = 0):
        """Play stream with specific settings"""
        # Use getattr to dynamically call the correct audio engine method
        getattr(self.audio, f'set_{self.stream_type}_pitch')(pitch)
        getattr(self.audio, f'set_{self.stream_type}_volume')(volume)
        getattr(self.audio, f'set_{self.stream_type}_trim')(start_time, end_time)
        
        return getattr(self.audio, f'play_{self.stream_type}')(url)
    
    def _stop(self):
        """Stop streaming"""
        getattr(self.audio, f'stop_{self.stream_type}')()
    
    def _pause(self):
        """Pause streaming"""
        getattr(self.audio, f'pause_{self.stream_type}')()
    
    def _resume(self):
        """Resume streaming"""
        getattr(self.audio, f'resume_{self.stream_type}')()
    
    def _get_info(self):
        """Get playback info"""
        return getattr(self.audio, f'get_{self.stream_type}_info')()
    
    def _is_playing(self):
        """Check if playing"""
        return getattr(self.audio, f'is_{self.stream_type}_playing')()
    
    def _set_volume(self, vol: float):
        """Set volume"""
        getattr(self.audio, f'set_{self.stream_type}_volume')(vol)
        return True
    
    def _get_items(self):
        """Get all cached items"""
        from core.config import load_sound_settings
        
        stream = self._get_stream_object()
        items = []
        
        settings = load_sound_settings()
        keybinds = settings.get(self._get_keybinds_key(), {})
        
        for key, data in stream._cache_index.items():
            items.append({
                'url': data['url'],
                'title': data['title'],
                'file': data['file'],
                'keybind': keybinds.get(data['url'], '')
            })
        
        return items
    
    def _add_item(self, url: str):
        """Add item (download and cache)"""
        def on_progress(d):
            if d['status'] == 'downloading':
                try:
                    percent_str = d.get('_percent_str', '').strip()
                    # Remove ANSI color codes using pre-compiled regex
                    percent_str = self.ANSI_ESCAPE_PATTERN.sub('', percent_str)
                    
                    percent = 0
                    if '%' in percent_str:
                        percent = float(percent_str.replace('%', ''))
                    
                    # Send to frontend - use dynamic function name
                    progress_func = f'on{self.stream_type.capitalize()}Progress'
                    getattr(eel, progress_func)(url, percent)
                    eel.sleep(0.01)
                except Exception as e:
                    # Log error instead of silent failure
                    print(f"[{self.stream_type.upper()}] Progress callback error: {e}")
        
        result = getattr(self.audio, f'play_{self.stream_type}')(url, on_progress)
        if result['success']:
            getattr(self.audio, f'stop_{self.stream_type}')()  # Stop after caching
        return result
    
    def _delete_item(self, url: str):
        """Delete cached item"""
        import os
        
        stream = self._get_stream_object()
        cached_file, title = stream._get_cached_file(url)
        
        if not cached_file:
            return {'success': False, 'error': 'Not found'}
        
        try:
            # Delete file
            os.remove(cached_file)
            
            # Remove from index
            cache_key = stream._get_cache_key(url)
            if cache_key in stream._cache_index:
                del stream._cache_index[cache_key]
                stream._save_cache_index()
            
            return {'success': True}
        except Exception as e:
            print(f"[{self.stream_type.upper()}] Delete error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_as_sound(self, url: str):
        """Save cache as a sound item"""
        import shutil
        from pathlib import Path
        
        stream = self._get_stream_object()
        cached_file, title = stream._get_cached_file(url)
        
        if not cached_file:
            return {'success': False, 'error': 'Not cached yet'}
        
        # Copy to sounds directory
        src = Path(cached_file)
        dest = Path(self.sounds_dir) / f"{title[:50]}{src.suffix}"
        
        try:
            shutil.copy(src, dest)
            self.audio.load_sounds()
            return {'success': True, 'name': dest.stem}
        except Exception as e:
            print(f"[{self.stream_type.upper()}] Save as sound error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_duration(self, url: str):
        """Get video duration in seconds"""
        return self._get_stream_object().get_duration(url)
