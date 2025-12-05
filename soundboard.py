"""
Core Soundboard Logic
"""
try:
    import pygame
    AUDIO_BACKEND = 'pygame'
except ImportError:
    try:
        import pygame_ce as pygame
        AUDIO_BACKEND = 'pygame_ce'
    except ImportError:
        pygame = None
        AUDIO_BACKEND = None

import os
from pathlib import Path

class Soundboard:
    def __init__(self, sounds_dir="sounds"):
        if pygame:
            pygame.mixer.init()
        self.sounds_dir = Path(sounds_dir)
        self.sounds = {}
        self.load_sounds()
    
    def load_sounds(self):
        """Load all audio files from sounds directory"""
        if not self.sounds_dir.exists():
            self.sounds_dir.mkdir(parents=True)
            return
        
        for file in self.sounds_dir.glob("*.wav"):
            name = file.stem
            self.sounds[name] = str(file)
    
    def play_sound(self, sound_name):
        """Play a sound by name"""
        if sound_name in self.sounds:
            sound = pygame.mixer.Sound(self.sounds[sound_name])
            sound.play()
            return True
        return False
    
    def stop_all(self):
        """Stop all playing sounds"""
        if pygame:
            pygame.mixer.stop()
    
    def set_volume(self, volume):
        """Set global volume (0.0 to 1.0)"""
        if pygame:
            pygame.mixer.music.set_volume(volume)
    
    def get_sound_list(self):
        """Get list of available sounds"""
        return sorted(list(self.sounds.keys()))
    
    def add_sound(self, file_path, name=None):
        """Add a new sound to the soundboard"""
        path = Path(file_path)
        if not path.exists():
            return False
        
        sound_name = name or path.stem
        dest = self.sounds_dir / f"{sound_name}{path.suffix}"
        
        if path != dest:
            import shutil
            shutil.copy(path, dest)
        
        self.sounds[sound_name] = str(dest)
        return True
