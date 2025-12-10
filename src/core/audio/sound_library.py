from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)

class SoundLibrary:
    """Manages the collection of sound files on disk."""
    
    def __init__(self, sounds_dir: str):
        self.sounds_dir = Path(sounds_dir)
        self.sounds: dict[str, str] = {}
        self.load_sounds()
        
    def load_sounds(self):
        """Load sounds from directory"""
        self.sounds.clear()
        if not self.sounds_dir.exists():
            self.sounds_dir.mkdir(parents=True, exist_ok=True)
            return
        
        for ext in ('*.wav', '*.mp3', '*.ogg', '*.flac'):
            for f in self.sounds_dir.glob(ext):
                self.sounds[f.stem] = str(f)
                
    def get_sounds(self) -> list[str]:
        """Get list of sound names"""
        return sorted(self.sounds.keys())
        
    def get_path(self, name: str) -> str:
        """Get path for a sound name"""
        return self.sounds.get(name)
        
    def add_sound(self, filepath: str, name: str = None) -> bool:
        """Add sound file to library"""
        src = Path(filepath)
        if not src.exists():
            return False
        
        dest_name = name or src.stem
        dest = self.sounds_dir / f"{dest_name}{src.suffix}"
        
        try:
            if src != dest:
                shutil.copy(src, dest)
            self.sounds[dest_name] = str(dest)
            return True
        except Exception as e:
            logger.debug(f"Failed to add sound {filepath}: {e}")
            return False
            
    def add_sound_from_data(self, filename: str, data: bytes) -> bool:
        """Add sound file from binary data"""
        try:
            # Get extension
            ext = Path(filename).suffix.lower()
            if ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
                return False
            
            # Save directly to sounds folder
            name = Path(filename).stem
            dest = self.sounds_dir / filename
            
            # Handle duplicate names
            counter = 1
            while dest.exists():
                dest = self.sounds_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            dest.write_bytes(data)
            self.load_sounds()  # Reload sounds
            
            # Update cache if needed or just let load_sounds handle it
            # self.sounds[dest.stem] = str(dest) # load_sounds does this
            return True
        except Exception as e:
            logger.debug(f"Error adding sound from data: {e}")
            return False

    def delete_sound(self, name: str) -> bool:
        """Delete sound from library"""
        if name not in self.sounds:
            return False
        try:
            Path(self.sounds[name]).unlink(missing_ok=True)
            del self.sounds[name]
            return True
        except Exception as e:
            logger.debug(f"Failed to delete sound {name}: {e}")
            return False
