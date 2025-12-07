"""API Layer - YouTube Endpoints"""
import eel


class YouTubeAPI:
    """YouTube-related API endpoints"""
    
    def __init__(self, audio_engine, sounds_dir):
        self.audio = audio_engine
        self.sounds_dir = sounds_dir
        self._register_endpoints()
    
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
    
    def play_youtube(self, url: str):
        """Play YouTube audio by URL"""
        from core.config import load_sound_settings
        
        # Load pitch and scream mode settings
        settings = load_sound_settings()
        pitch_mode_map = settings.get('youtubePitchMode', {})
        scream_mode_map = settings.get('youtubeScreamMode', {})
        
        # Handle backward compatibility
        if isinstance(pitch_mode_map, bool):
            pitch_mode_map = {}
        if isinstance(scream_mode_map, bool):
            scream_mode_map = {}
        
        # Apply settings
        pitch = 1.5 if pitch_mode_map.get(url, False) else 1.0
        vol = 50.0 if scream_mode_map.get(url, False) else 1.0
        
        self.audio.set_youtube_pitch(pitch)
        self.audio.set_youtube_volume(vol)
        
        return self.audio.play_youtube(url)
    
    def stop_youtube(self):
        """Stop YouTube streaming"""
        self.audio.stop_youtube()
    
    def pause_youtube(self):
        """Pause YouTube streaming"""
        self.audio.pause_youtube()
    
    def resume_youtube(self):
        """Resume YouTube streaming"""
        self.audio.resume_youtube()
    
    def get_youtube_info(self):
        """Get YouTube playback info"""
        return self.audio.get_youtube_info()
    
    def is_youtube_playing(self):
        """Check if YouTube is playing"""
        return self.audio.is_youtube_playing()
    
    def set_youtube_volume(self, vol: float):
        """Set YouTube volume"""
        self.audio.set_youtube_volume(vol)
        return True
    
    def get_youtube_items(self):
        """Get all YouTube cached items"""
        from core.config import load_sound_settings
        
        yt_stream = self.audio.youtube
        items = []
        
        settings = load_sound_settings()
        youtube_keybinds = settings.get('youtubeKeybinds', {})
        
        for key, data in yt_stream._cache_index.items():
            items.append({
                'url': data['url'],
                'title': data['title'],
                'file': data['file'],
                'keybind': youtube_keybinds.get(data['url'], '')
            })
        
        return items
    
    def add_youtube_item(self, url: str):
        """Add YouTube item (download and cache)"""
        def on_progress(d):
            if d['status'] == 'downloading':
                try:
                    percent_str = d.get('_percent_str', '').strip()
                    # Remove ANSI color codes
                    import re
                    percent_str = re.sub(r'\x1b\[[0-9;]*m', '', percent_str)
                    
                    percent = 0
                    if '%' in percent_str:
                        percent = float(percent_str.replace('%', ''))
                    
                    # Send to frontend
                    eel.onYoutubeProgress(url, percent)
                    eel.sleep(0.01)
                except Exception:
                    pass
        
        result = self.audio.play_youtube(url, on_progress)
        if result['success']:
            self.audio.stop_youtube()  # Stop after caching
        return result
    
    def delete_youtube_item(self, url: str):
        """Delete YouTube cached item"""
        import os
        
        yt_stream = self.audio.youtube
        cached_file, title = yt_stream._get_cached_file(url)
        
        if not cached_file:
            return {'success': False, 'error': 'Not found'}
        
        try:
            # Delete file
            os.remove(cached_file)
            
            # Remove from index
            cache_key = yt_stream._get_cache_key(url)
            if cache_key in yt_stream._cache_index:
                del yt_stream._cache_index[cache_key]
                yt_stream._save_cache_index()
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_youtube_as_sound(self, url: str):
        """Save YouTube cache as a sound item"""
        import shutil
        from pathlib import Path
        
        yt_stream = self.audio.youtube
        cached_file, title = yt_stream._get_cached_file(url)
        
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
            return {'success': False, 'error': str(e)}
