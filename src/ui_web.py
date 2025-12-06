"""
Soundboard Web UI using Eel (HTML/CSS/JS frontend)
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eel
from soundboard import Soundboard
from config import Config

# Initialize
config = Config()
soundboard = Soundboard(config.sounds_dir)
soundboard.set_volume(config.default_volume)

# Settings file for keybinds and volumes
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sound_settings.json')

# Get web folder path
web_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
eel.init(web_folder)


# ===== Exposed functions for JavaScript =====

@eel.expose
def get_sounds():
    """Get list of available sounds"""
    return soundboard.get_sound_list()


@eel.expose
def play_sound(name, volume=1.0):
    """Play a sound by name with specific volume"""
    # Temporarily set volume for this sound
    original_volume = soundboard.volume
    soundboard.set_volume(volume)
    result = soundboard.play_sound(name)
    # Note: volume will be applied to this play
    return result


@eel.expose
def stop_all():
    """Stop all playing sounds"""
    soundboard.stop_all()
    return True


@eel.expose
def set_volume(volume):
    """Set global volume (0.0 to 1.0)"""
    soundboard.set_volume(volume)
    return True


@eel.expose
def get_volume():
    """Get current volume"""
    return soundboard.volume


@eel.expose
def is_vb_cable_connected():
    """Check if VB-Cable is connected"""
    return soundboard.is_vb_cable_connected()


@eel.expose
def add_sound_dialog():
    """Open file dialog to add sound"""
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_paths = filedialog.askopenfilenames(
        title="Select Audio Files",
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.ogg *.flac"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    
    added = 0
    for file_path in file_paths:
        if soundboard.add_sound(file_path):
            added += 1
    
    return added > 0


@eel.expose
def delete_sound(name):
    """Delete a sound"""
    if name in soundboard.sounds:
        try:
            file_path = soundboard.sounds[name]
            if os.path.exists(file_path):
                os.remove(file_path)
            del soundboard.sounds[name]
            return True
        except Exception as e:
            print(f"Error deleting sound: {e}")
    return False


@eel.expose
def get_settings():
    """Get saved settings (volumes, keybinds)"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    return {'volumes': {}, 'keybinds': {}}


@eel.expose
def save_settings(settings):
    """Save settings (volumes, keybinds)"""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False


def on_close(page, sockets):
    """Handle window close"""
    if hasattr(soundboard, 'cleanup'):
        soundboard.cleanup()
    sys.exit()


def main():
    """Start the web UI"""
    print("🎵 Starting Soundboard Pro...")
    
    try:
        eel.start(
            'index.html',
            size=(1100, 750),
            position=(100, 100),
            close_callback=on_close,
            mode='chrome',
            cmdline_args=['--disable-gpu']
        )
    except EnvironmentError:
        try:
            eel.start(
                'index.html',
                size=(1100, 750),
                position=(100, 100),
                close_callback=on_close,
                mode='edge'
            )
        except EnvironmentError:
            print("⚠️ Chrome/Edge not found, opening in default browser...")
            eel.start(
                'index.html',
                size=(1100, 750),
                close_callback=on_close,
                mode='default'
            )


if __name__ == "__main__":
    main()
