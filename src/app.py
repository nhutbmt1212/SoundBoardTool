"""Soundboard Pro - Web UI Application"""
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eel
from core import AudioEngine, Config
from core.config import load_sound_settings, save_sound_settings

# Initialize
config = Config()
audio = AudioEngine(config.sounds_dir)
audio.set_volume(config.default_volume)

# Web folder
WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')
eel.init(WEB_DIR)


# === API Functions ===

@eel.expose
def get_sounds():
    return audio.get_sounds()


@eel.expose
def play_sound(name: str, volume: float = 1.0):
    audio.set_volume(volume)
    return audio.play(name)


@eel.expose
def stop_all():
    audio.stop()
    return True


@eel.expose
def set_volume(vol: float):
    audio.set_volume(vol)
    return True


@eel.expose
def get_volume():
    return audio.volume


@eel.expose
def is_vb_cable_connected():
    return audio.is_vb_connected()


@eel.expose
def add_sound_dialog():
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
    
    added = sum(1 for f in files if audio.add_sound(f))
    return added > 0


@eel.expose
def delete_sound(name: str):
    return audio.delete_sound(name)


@eel.expose
def get_settings():
    return load_sound_settings()


@eel.expose
def save_settings(settings: dict):
    save_sound_settings(settings)
    return True


def on_close(page, sockets):
    audio.cleanup()
    sys.exit()


def main():
    print("🎵 Soundboard Pro")
    
    try:
        eel.start('index.html', size=(1100, 750), close_callback=on_close, mode='chrome')
    except EnvironmentError:
        try:
            eel.start('index.html', size=(1100, 750), close_callback=on_close, mode='edge')
        except EnvironmentError:
            eel.start('index.html', size=(1100, 750), close_callback=on_close, mode='default')


if __name__ == "__main__":
    main()
