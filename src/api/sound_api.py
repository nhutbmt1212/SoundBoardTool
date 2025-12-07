"""API Layer - Sound Endpoints"""
import eel
from pathlib import Path


class SoundAPI:
    """Sound-related API endpoints"""
    
    def __init__(self, audio_engine, sounds_dir):
        self.audio = audio_engine
        self.sounds_dir = sounds_dir
        self._register_endpoints()
    
    def _register_endpoints(self):
        """Register all sound endpoints with Eel"""
        eel.expose(self.get_sounds)
        eel.expose(self.play_sound)
        eel.expose(self.stop_all)
        eel.expose(self.set_volume)
        eel.expose(self.get_volume)
        eel.expose(self.get_playing_sound)
        eel.expose(self.add_sound_dialog)
        eel.expose(self.add_sound_base64)
        eel.expose(self.delete_sound)
    
    def get_sounds(self):
        """Get list of available sounds"""
        return self.audio.get_sounds()
    
    def play_sound(self, name: str, volume: float = 1.0, pitch: float = 1.0):
        """Play sound with volume and pitch"""
        self.audio.set_volume(volume)
        self.audio.set_pitch(pitch)
        return self.audio.play(name)
    
    def stop_all(self):
        """Stop all playing sounds"""
        self.audio.stop()
        return True
    
    def set_volume(self, vol: float):
        """Set playback volume"""
        self.audio.set_volume(vol)
        return True
    
    def get_volume(self):
        """Get current volume"""
        return self.audio.volume
    
    def get_playing_sound(self):
        """Get currently playing sound name"""
        return self.audio._current_playing_sound
    
    def add_sound_dialog(self):
        """Open file dialog to add sounds"""
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac"), ("All", "*.*")]
        )
        root.destroy()
        
        added = sum(1 for f in files if self.audio.add_sound(f))
        return added > 0
    
    def add_sound_base64(self, filename: str, base64_data: str):
        """Add sound from base64 data (for drag & drop)"""
        import base64
        
        try:
            # Decode base64
            data = base64.b64decode(base64_data)
            
            # Get extension
            ext = Path(filename).suffix.lower()
            if ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
                return False
            
            # Save directly to sounds folder
            name = Path(filename).stem
            dest = Path(self.sounds_dir) / filename
            
            # Handle duplicate names
            counter = 1
            while dest.exists():
                dest = Path(self.sounds_dir) / f"{name}_{counter}{ext}"
                counter += 1
            
            dest.write_bytes(data)
            self.audio.load_sounds()  # Reload sounds
            return True
        except Exception as e:
            print(f"Error adding sound: {e}")
            return False
    
    def delete_sound(self, name: str):
        """Delete a sound"""
        return self.audio.delete_sound(name)
