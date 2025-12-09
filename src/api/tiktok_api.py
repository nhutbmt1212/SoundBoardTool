"""API Layer - TikTok Endpoints"""
import eel


class TikTokAPI:
    """TikTok-related API endpoints"""
    
    def __init__(self, audio_engine, sounds_dir):
        self.audio = audio_engine
        self.sounds_dir = sounds_dir
        self._register_endpoints()
    
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
    
    def play_tiktok(self, url: str, volume: float = 1.0, pitch: float = 1.0, start_time: float = 0, end_time: float = 0):
        """Play TikTok audio by URL with specific settings"""
        
        self.audio.set_tiktok_pitch(pitch)
        self.audio.set_tiktok_volume(volume)
        self.audio.set_tiktok_trim(start_time, end_time)
        
        return self.audio.play_tiktok(url)
    
    def stop_tiktok(self):
        """Stop TikTok streaming"""
        self.audio.stop_tiktok()
    
    def pause_tiktok(self):
        """Pause TikTok streaming"""
        self.audio.pause_tiktok()
    
    def resume_tiktok(self):
        """Resume TikTok streaming"""
        self.audio.resume_tiktok()
    
    def get_tiktok_info(self):
        """Get TikTok playback info"""
        return self.audio.get_tiktok_info()
    
    def is_tiktok_playing(self):
        """Check if TikTok is playing"""
        return self.audio.is_tiktok_playing()
    
    def set_tiktok_volume(self, vol: float):
        """Set TikTok volume"""
        self.audio.set_tiktok_volume(vol)
        return True
    
    def get_tiktok_items(self):
        """Get all TikTok cached items"""
        from core.config import load_sound_settings
        
        tiktok_stream = self.audio.tiktok
        items = []
        
        settings = load_sound_settings()
        tiktok_keybinds = settings.get('tiktokKeybinds', {})
        
        for key, data in tiktok_stream._cache_index.items():
            items.append({
                'url': data['url'],
                'title': data['title'],
                'file': data['file'],
                'keybind': tiktok_keybinds.get(data['url'], '')
            })
        
        return items
    
    def add_tiktok_item(self, url: str):
        """Add TikTok item (download and cache)"""
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
                    eel.onTiktokProgress(url, percent)
                    eel.sleep(0.01)
                except Exception:
                    pass
        
        result = self.audio.play_tiktok(url, on_progress)
        if result['success']:
            self.audio.stop_tiktok()  # Stop after caching
        return result
    
    def delete_tiktok_item(self, url: str):
        """Delete TikTok cached item"""
        import os
        
        tiktok_stream = self.audio.tiktok
        cached_file, title = tiktok_stream._get_cached_file(url)
        
        if not cached_file:
            return {'success': False, 'error': 'Not found'}
        
        try:
            # Delete file
            os.remove(cached_file)
            
            # Remove from index
            cache_key = tiktok_stream._get_cache_key(url)
            if cache_key in tiktok_stream._cache_index:
                del tiktok_stream._cache_index[cache_key]
                tiktok_stream._save_cache_index()
            
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def save_tiktok_as_sound(self, url: str):
        """Save TikTok cache as a sound item"""
        import shutil
        from pathlib import Path
        
        tiktok_stream = self.audio.tiktok
        cached_file, title = tiktok_stream._get_cached_file(url)
        
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
    
    def get_tiktok_duration(self, url: str):
        """Get TikTok video duration in seconds"""
        return self.audio.tiktok.get_duration(url)
