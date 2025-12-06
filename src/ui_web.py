"""
Soundboard Web UI using Eel (HTML/CSS/JS frontend)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eel
from soundboard import Soundboard
from config import Config

# Initialize
config = Config()
soundboard = Soundboard(config.sounds_dir)
soundboard.set_volume(config.default_volume)

# Get web folder path
web_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
eel.init(web_folder)


# ===== Exposed functions for JavaScript =====

@eel.expose
def get_sounds():
    """Get list of available sounds"""
    return soundboard.get_sound_list()


@eel.expose
def play_sound(name):
    """Play a sound by name"""
    return soundboard.play_sound(name)


@eel.expose
def stop_all():
    """Stop all playing sounds"""
    soundboard.stop_all()
    return True


@eel.expose
def set_volume(volume):
    """Set volume (0.0 to 1.0)"""
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
    
    # Hide main tkinter window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.ogg *.flac"),
            ("All Files", "*.*")
        ]
    )
    
    root.destroy()
    
    if file_path:
        return soundboard.add_sound(file_path)
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
        # Start Eel with Chrome/Edge in app mode
        eel.start(
            'index.html',
            size=(950, 750),
            position=(100, 100),
            close_callback=on_close,
            mode='chrome',  # Will fallback to edge or default browser
            cmdline_args=['--disable-gpu']  # Fix for some systems
        )
    except EnvironmentError:
        # If Chrome not found, try edge
        try:
            eel.start(
                'index.html',
                size=(950, 750),
                position=(100, 100),
                close_callback=on_close,
                mode='edge'
            )
        except EnvironmentError:
            # Fallback to default browser
            print("⚠️ Chrome/Edge not found, opening in default browser...")
            eel.start(
                'index.html',
                size=(950, 750),
                close_callback=on_close,
                mode='default'
            )


if __name__ == "__main__":
    main()
