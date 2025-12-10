"""Hotkey Service - Manages global hotkeys"""
from core.config import load_sound_settings


class HotkeyService:
    """Manages global hotkey registration and callbacks"""
    
    def __init__(self):
        self.manager = None
        self.sound_callback = None
        self.youtube_callback = None
        self.tiktok_callback = None
        self.stop_callback = None
        
        # Settings cache
        self.sound_volumes = {}
        self.sound_scream_mode = {}
        self.sound_pitch_mode = {}
        self.sound_loop_state = {}
        self.youtube_volumes = {}
        self.youtube_scream_mode = {}
        self.youtube_pitch_mode = {}
        self.youtube_loop_state = {}
        self.tiktok_volumes = {}
        self.tiktok_scream_mode = {}
        self.tiktok_pitch_mode = {}
        self.tiktok_pitch_mode = {}
        self.tiktok_loop_state = {}
        
        # Effects cache
        self.sound_effects = {}
        self.youtube_effects = {}
        self.tiktok_effects = {}
    
    def initialize(self, sound_callback, youtube_callback, tiktok_callback, stop_callback):
        """Initialize hotkey service with callbacks"""
        self.sound_callback = sound_callback
        self.youtube_callback = youtube_callback
        self.tiktok_callback = tiktok_callback
        self.stop_callback = stop_callback
        
        # Lazy import hotkey manager
        try:
            from core.hotkey import hotkey_manager
            self.manager = hotkey_manager
        except Exception as e:
            print(f"Hotkey manager not available: {e}")
    
    def update_from_settings(self):
        """Update all hotkeys from settings"""
        settings = load_sound_settings()
        
        # Defensive check
        if settings is None:
            settings = {'volumes': {}, 'keybinds': {}}
        
        # Update caches
        self.sound_volumes = settings.get('volumes', {})
        self.sound_scream_mode = settings.get('screamMode', {})
        self.sound_pitch_mode = settings.get('pitchMode', {})
        self.sound_loop_state = settings.get('soundLoopState', {})
        self.youtube_volumes = settings.get('youtubeVolumes', {})
        self.youtube_scream_mode = settings.get('youtubeScreamMode', {})
        self.youtube_pitch_mode = settings.get('youtubePitchMode', {})
        self.youtube_loop_state = settings.get('youtubeLoopState', {})
        self.tiktok_volumes = settings.get('tiktokVolumes', {})
        self.tiktok_scream_mode = settings.get('tiktokScreamMode', {})
        self.tiktok_pitch_mode = settings.get('tiktokPitchMode', {})
        self.tiktok_pitch_mode = settings.get('tiktokPitchMode', {})
        self.tiktok_loop_state = settings.get('tiktokLoopState', {})
        
        # Update effects
        self.sound_effects = settings.get('soundEffects', {})
        self.youtube_effects = settings.get('youtubeEffects', {})
        self.tiktok_effects = settings.get('tiktokEffects', {})
        
        # Backward compatibility
        if isinstance(self.youtube_scream_mode, bool):
            self.youtube_scream_mode = {}
        if isinstance(self.youtube_pitch_mode, bool):
            self.youtube_pitch_mode = {}
        
        # Register hotkeys
        if self.manager:
            keybinds = settings.get('keybinds', {})
            youtube_keybinds = settings.get('youtubeKeybinds', {})
            tiktok_keybinds = settings.get('tiktokKeybinds', {})
            stop_keybind = settings.get('stopAllKeybind', '')
            
            self.manager.update_all(
                keybinds,
                self.sound_callback,
                self.stop_callback,
                stop_keybind,
                youtube_keybinds,
                self.youtube_callback,
                tiktok_keybinds,
                self.tiktok_callback
            )
    
    def get_sound_settings(self, name: str):
        """Get cached settings for a sound"""
        return {
            'volume': self.sound_volumes.get(name, 100),
            'scream': self.sound_scream_mode.get(name, False),
            'pitch': self.sound_pitch_mode.get(name, False),
            'pitch': self.sound_pitch_mode.get(name, False),
            'loop': self.sound_loop_state.get(name, False),
            'effects': self.sound_effects.get(name, {})
        }
    
    def get_youtube_settings(self, url: str):
        """Get cached settings for a YouTube URL"""
        return {
            'volume': self.youtube_volumes.get(url, 100),
            'scream': self.youtube_scream_mode.get(url, False),
            'pitch': self.youtube_pitch_mode.get(url, False),
            'pitch': self.youtube_pitch_mode.get(url, False),
            'loop': self.youtube_loop_state.get(url, False),
            'effects': self.youtube_effects.get(url, {})
        }
    
    def get_tiktok_settings(self, url: str):
        """Get cached settings for a TikTok URL"""
        return {
            'volume': self.tiktok_volumes.get(url, 100),
            'scream': self.tiktok_scream_mode.get(url, False),
            'pitch': self.tiktok_pitch_mode.get(url, False),
            'pitch': self.tiktok_pitch_mode.get(url, False),
            'loop': self.tiktok_loop_state.get(url, False),
            'effects': self.tiktok_effects.get(url, {})
        }
    
    def cleanup(self):
        """Cleanup hotkeys"""
        if self.manager:
            try:
                self.manager.unregister_all()
            except Exception:
                pass
