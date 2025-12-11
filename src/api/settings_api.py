"""API Layer - Settings and Microphone Endpoints"""
import eel
from core.config import load_sound_settings, save_sound_settings


class SettingsAPI:
    """Settings and microphone-related API endpoints"""
    
    def __init__(self, audio_engine, hotkey_update_callback):
        self.audio = audio_engine
        self.hotkey_update_callback = hotkey_update_callback
        self._register_endpoints()
    
    def _register_endpoints(self):
        """Register all settings endpoints with Eel"""
        eel.expose(self.get_settings)
        eel.expose(self.save_settings)
        eel.expose(self.is_vb_cable_connected)
        eel.expose(self.get_mic_devices)
        eel.expose(self.set_mic_device)
        eel.expose(self.set_mic_volume)
        eel.expose(self.toggle_mic_passthrough)
        eel.expose(self.is_mic_enabled)
    
    def get_settings(self):
        """Get application settings"""
        return load_sound_settings()
    
    def save_settings(self, settings: dict):
        """Save application settings"""
        save_sound_settings(settings)
        # Update global hotkeys when settings change
        if self.hotkey_update_callback:
            self.hotkey_update_callback()
        
        # Trigger auto backup if enabled
        try:
            from .backup_api import trigger_auto_backup
            trigger_auto_backup()
        except Exception as e:
            print(f"[SettingsAPI] Auto backup failed: {e}")
        
        return True
    
    def is_vb_cable_connected(self):
        """Check if VB-Cable is connected"""
        return self.audio.is_vb_connected()
    
    def get_mic_devices(self):
        """Get list of microphone devices"""
        return self.audio.get_mic_devices()
    
    def set_mic_device(self, device_id: int):
        """Set microphone device"""
        self.audio.set_mic_device(device_id)
        return True
    
    def set_mic_volume(self, vol: float):
        """Set microphone volume"""
        self.audio.set_mic_volume(vol)
        return True
    
    def toggle_mic_passthrough(self, enabled: bool):
        """Toggle microphone passthrough"""
        if enabled:
            return self.audio.start_mic_passthrough()
        else:
            self.audio.stop_mic_passthrough()
            return True
    
    def is_mic_enabled(self):
        """Check if microphone is enabled"""
        return self.audio.is_mic_enabled()
