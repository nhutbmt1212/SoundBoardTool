"""Configuration management"""
import json
from pathlib import Path
from .config_paths import get_config_path, get_sounds_dir

# Use AppData for config files
CONFIG_FILE = Path(get_config_path("soundboard_config.json"))
SETTINGS_FILE = Path(get_config_path("sound_settings.json"))


class Config:
    def __init__(self):
        self.sounds_dir = get_sounds_dir()
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

