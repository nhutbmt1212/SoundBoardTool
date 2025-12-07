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
sound_pitch_mode = {}

# Init eel
eel.init(WEB_DIR)


def play_sound_global(name: str):
    """Play sound from global hotkey"""
    vol = sound_volumes.get(name, 100) / 100
    # Apply scream mode (5000% boost)
    if sound_scream_mode.get(name, False):
        vol = min(vol * 50.0, 50.0)
    # Apply pitch mode (chipmunk)
    pitch = 1.5 if sound_pitch_mode.get(name, False) else 1.0
    audio.set_volume(vol)
    audio.set_pitch(pitch)
    audio.play(name)


def play_youtube_global(url: str):
    """Play YouTube from global hotkey"""
    audio.play_youtube(url)


def stop_all_global():
    """Stop all from global hotkey"""
    audio.stop()


def update_global_hotkeys():
    """Update global hotkeys from settings"""
    settings = load_sound_settings()
    keybinds = settings.get('keybinds', {})
    youtube_keybinds = settings.get('youtubeKeybinds', {})
    stop_keybind = settings.get('stopAllKeybind', '')
    
    # Update caches
    global sound_volumes, sound_scream_mode, sound_pitch_mode
    sound_volumes = settings.get('volumes', {})
    sound_scream_mode = settings.get('screamMode', {})
    sound_pitch_mode = settings.get('pitchMode', {})
    
    # Register hotkeys
    hm = get_hotkey_manager()
    if hm:
        hm.update_all(keybinds, play_sound_global, stop_all_global, stop_keybind, 
                      youtube_keybinds, play_youtube_global)


# === API Functions ===

@eel.expose
def get_sounds():
    return audio.get_sounds()


@eel.expose
def play_sound(name: str, volume: float = 1.0, pitch: float = 1.0):
    audio.set_volume(volume)
    audio.set_pitch(pitch)
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
def get_playing_sound():
    """Get currently playing sound name"""
    return audio._current_playing_sound


@eel.expose
def is_vb_cable_connected():
    return audio.is_vb_connected()


@eel.expose
def get_mic_devices():
    return audio.get_mic_devices()


@eel.expose
def set_mic_device(device_id: int):
    audio.set_mic_device(device_id)
    return True


@eel.expose
def set_mic_volume(vol: float):
    audio.set_mic_volume(vol)
    return True


@eel.expose
def toggle_mic_passthrough(enabled: bool):
    if enabled:
        return audio.start_mic_passthrough()
    else:
        audio.stop_mic_passthrough()
        return True


@eel.expose
def is_mic_enabled():
    return audio.is_mic_enabled()


# === YouTube Streaming ===

@eel.expose
def play_youtube(url: str):
    """Play YouTube audio by URL"""
    return audio.play_youtube(url)


@eel.expose
def stop_youtube():
    """Stop YouTube streaming"""
    audio.stop_youtube()


@eel.expose
def pause_youtube():
    """Pause YouTube streaming"""
    audio.pause_youtube()


@eel.expose
def resume_youtube():
    """Resume YouTube streaming"""
    audio.resume_youtube()


@eel.expose
def save_youtube_as_sound(url: str):
    """Save YouTube cache as a sound item"""
    import shutil
    from pathlib import Path
    
    # Get cached file from YouTube
    yt_stream = audio.youtube
    cached_file, title = yt_stream._get_cached_file(url)
    
    if not cached_file:
        return {'success': False, 'error': 'Not cached yet'}
    
    # Copy to sounds directory
    src = Path(cached_file)
    dest = Path(sounds_dir) / f"{title[:50]}{src.suffix}"  # Limit filename length
    
    try:
        shutil.copy(src, dest)
        audio.load_sounds()  # Reload sounds
        return {'success': True, 'name': dest.stem}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@eel.expose
def get_youtube_items():
    """Get all YouTube cached items"""
    yt_stream = audio.youtube
    items = []
    
    settings = load_sound_settings()
    youtube_keybinds = settings.get('youtubeKeybinds', {})
    
    for key, data in yt_stream._cache_index.items():
        items.append({
            'url': data['url'],
            'title': data['title'],
            'file': data['file'],
            'keybind': youtube_keybinds.get(data['url'], '')
        })
    
    return items


@eel.expose
def add_youtube_item(url: str):
    """Add YouTube item (download and cache)"""
    result = audio.play_youtube(url)
    if result['success']:
        audio.stop_youtube()  # Stop after caching
    return result


@eel.expose
def delete_youtube_item(url: str):
    """Delete YouTube cached item"""
    import os
    from pathlib import Path
    
    yt_stream = audio.youtube
    cached_file, title = yt_stream._get_cached_file(url)
    
    if not cached_file:
        return {'success': False, 'error': 'Not found'}
    
    try:
        # Delete file
        os.remove(cached_file)
        
        # Remove from index
        cache_key = yt_stream._get_cache_key(url)
        if cache_key in yt_stream._cache_index:
            del yt_stream._cache_index[cache_key]
            yt_stream._save_cache_index()
        
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    return True


@eel.expose
def is_youtube_playing():
    return audio.is_youtube_playing()


@eel.expose
def get_youtube_info():
    return audio.get_youtube_info()


@eel.expose
def set_youtube_volume(vol: float):
    audio.set_youtube_volume(vol)
    return True


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
def add_sounds_by_path(paths: list):
    """Add sounds by file paths (for drag & drop)"""
    added = 0
    for path in paths:
        if path and audio.add_sound(path):
            added += 1
    return added


@eel.expose
def add_sound_base64(filename: str, base64_data: str):
    """Add sound from base64 data (for drag & drop from browser)"""
    import base64
    import tempfile
    from pathlib import Path
    
    try:
        # Decode base64
        data = base64.b64decode(base64_data)
        
        # Get extension
        ext = Path(filename).suffix.lower()
        if ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a']:
            return False
        
        # Save directly to sounds folder
        name = Path(filename).stem
        dest = Path(sounds_dir) / filename
        
        # Handle duplicate names
        counter = 1
        while dest.exists():
            dest = Path(sounds_dir) / f"{name}_{counter}{ext}"
            counter += 1
        
        dest.write_bytes(data)
        audio.load_sounds()  # Reload sounds
        return True
    except Exception as e:
        print(f"Error adding sound: {e}")
        return False
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
        'cmdline_args': ['--disable-extensions']
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
            # Use specific browser args to disable extensions
            eel.start('index.html', size=(1100, 750), mode='default', 
                     cmdline_args=['--disable-extensions'],
                     close_callback=on_close, block=False)
            track_browser_process()
            while True:
                eel.sleep(1.0)
        except Exception as e2:
            print(f"Failed: {e2}")
    finally:
        cleanup_and_exit()


if __name__ == "__main__":
    main()
