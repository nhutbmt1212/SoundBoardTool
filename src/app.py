"""Soundboard Pro - Web UI Application"""
import sys
import os

# Handle PyInstaller frozen app
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    # For PyInstaller, web folder is in _MEIPASS
    WEB_DIR = os.path.join(sys._MEIPASS, 'web')
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEB_DIR = os.path.join(os.path.dirname(__file__), 'web')

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Single instance check
from core.single_instance import acquire_lock, release_lock, show_existing_window

if not acquire_lock():
    print("Another instance is already running!")
    show_existing_window()
    sys.exit(0)

import eel
from core.audio import AudioEngine
from core.config import Config
from core.config import load_sound_settings, save_sound_settings

# Lazy import hotkey manager to avoid blocking
hotkey_manager = None

def get_hotkey_manager():
    global hotkey_manager
    if hotkey_manager is None:
        try:
            from core.hotkey import hotkey_manager as hm
            hotkey_manager = hm
        except Exception:
            pass
    return hotkey_manager

# Initialize with correct sounds path
sounds_dir = os.path.join(BASE_DIR, 'sounds')
os.makedirs(sounds_dir, exist_ok=True)

config = Config()
audio = AudioEngine(sounds_dir)
audio.set_volume(config.default_volume)

# Sound settings cache for global hotkeys
sound_volumes = {}
sound_scream_mode = {}

# Init eel
eel.init(WEB_DIR)


def play_sound_global(name: str):
    """Play sound from global hotkey"""
    vol = sound_volumes.get(name, 100) / 100
    # Apply scream mode (500% boost)
    if sound_scream_mode.get(name, False):
        vol = min(vol * 5.0, 5.0)
    audio.set_volume(vol)
    audio.play(name)


def stop_all_global():
    """Stop all from global hotkey"""
    audio.stop()


def update_global_hotkeys():
    """Update global hotkeys from settings"""
    settings = load_sound_settings()
    keybinds = settings.get('keybinds', {})
    stop_keybind = settings.get('stopAllKeybind', '')
    
    # Update caches
    global sound_volumes, sound_scream_mode
    sound_volumes = settings.get('volumes', {})
    sound_scream_mode = settings.get('screamMode', {})
    
    # Register hotkeys
    hm = get_hotkey_manager()
    if hm:
        hm.update_all(keybinds, play_sound_global, stop_all_global, stop_keybind)


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
    # Update global hotkeys when settings change
    update_global_hotkeys()
    return True


def on_close(page, sockets):
    cleanup_and_exit()


def cleanup_and_exit():
    """Clean up all resources and exit"""
    from core.single_instance import kill_browser
    
    # Kill browser first
    kill_browser()
    
    hm = get_hotkey_manager()
    if hm:
        try:
            hm.unregister_all()
        except Exception:
            pass
    
    try:
        audio.cleanup()
    except Exception:
        pass
    
    release_lock()
    os._exit(0)


def find_browser():
    """Find available browser"""
    # Check Chrome
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in chrome_paths:
        if os.path.exists(path):
            return 'chrome', path
    
    # Check Edge (always available on Windows 10/11)
    edge_paths = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in edge_paths:
        if os.path.exists(path):
            return 'edge', path
    
    return None, None


def track_browser_process():
    """Find and track browser process spawned by eel"""
    import time
    from core.single_instance import set_browser_pid
    
    time.sleep(0.5)  # Wait for browser to spawn
    
    try:
        import psutil
        current_pid = os.getpid()
        
        # Find browser child process
        for proc in psutil.process_iter(['pid', 'name', 'ppid']):
            try:
                name = proc.info['name'].lower()
                if 'chrome' in name or 'msedge' in name:
                    # Check if it's our child or recently spawned
                    if proc.info['ppid'] == current_pid:
                        set_browser_pid(proc.info['pid'])
                        return
            except Exception:
                continue
        
        # Fallback: find by command line
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                name = proc.info['name'].lower()
                cmdline = proc.info.get('cmdline', [])
                if ('chrome' in name or 'msedge' in name) and cmdline:
                    cmd_str = ' '.join(cmdline).lower()
                    if '--app=' in cmd_str:
                        set_browser_pid(proc.info['pid'])
                        return
            except Exception:
                continue
    except ImportError:
        # psutil not available, use tasklist fallback
        pass


def main():
    import signal
    
    # Handle Ctrl+C and terminal close
    def signal_handler(sig, frame):
        print("\n   Shutting down...")
        cleanup_and_exit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🎵 Soundboard Pro")
    print(f"   Sounds: {sounds_dir}")
    
    # Check and install VB-Cable if needed
    try:
        from core.vbcable import ensure_vbcable
        if not audio.is_vb_connected():
            if ensure_vbcable():
                # Re-detect after install
                audio._detect_vb_cable()
    except Exception as e:
        print(f"   VB-Cable check: {e}")
    
    # Load and register global hotkeys (in try block to not block app)
    try:
        update_global_hotkeys()
        print("   Hotkeys: loaded")
    except Exception as e:
        print(f"   Hotkeys: failed ({e})")
    
    # Find browser
    browser_mode, browser_path = find_browser()
    
    eel_options = {
        'size': (1100, 750),
        'close_callback': on_close,
        'port': 0,  # Auto port
    }
    
    if browser_mode:
        eel_options['mode'] = browser_mode
        print(f"   Browser: {browser_mode}")
    else:
        eel_options['mode'] = 'default'
        print("   Browser: default")
    
    print("   Starting...")
    
    try:
        # Start eel - it spawns browser as subprocess
        eel.start('index.html', **eel_options, block=False)
        
        # Track browser PID after eel starts
        track_browser_process()
        
        # Keep main thread alive
        while True:
            eel.sleep(1.0)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\n   Interrupted")
    except Exception as e:
        print(f"Error: {e}")
        try:
            eel.start('index.html', size=(1100, 750), mode='default', close_callback=on_close, block=False)
            track_browser_process()
            while True:
                eel.sleep(1.0)
        except Exception as e2:
            print(f"Failed: {e2}")
    finally:
        cleanup_and_exit()


if __name__ == "__main__":
    main()
