"""Configuration management"""
import json
import sys
import os
from pathlib import Path

# Get base directory (works for both dev and frozen exe)
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(os.path.dirname(sys.executable))
else:
    BASE_DIR = Path(__file__).parent.parent.parent

CONFIG_FILE = BASE_DIR / "soundboard_config.json"
SETTINGS_FILE = BASE_DIR / "sound_settings.json"


class Config:
    def __init__(self):
        self.sounds_dir = str(BASE_DIR / "sounds")
        self.default_volume = 0.7
        self._load()
    
    def _load(self):
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
                self.default_volume = data.get('default_volume', self.default_volume)
            except Exception:
                pass
    
    def save(self):
        try:
            CONFIG_FILE.write_text(json.dumps({
                'sounds_dir': self.sounds_dir,
                'default_volume': self.default_volume,
            }, indent=2), encoding='utf-8')
        except Exception:
            pass


def load_sound_settings() -> dict:
    """Load per-sound settings (volumes, keybinds)"""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'volumes': {}, 'keybinds': {}}


def save_sound_settings(settings: dict):
    """Save per-sound settings"""
    try:
        SETTINGS_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    except Exception:
        pass
