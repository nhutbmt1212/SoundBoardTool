"""Soundboard Pro - Web UI Application (Refactored)"""
import sys
import os
import signal

# Handle PyInstaller frozen app
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
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
from api import SoundAPI, YouTubeAPI, SettingsAPI
from services import HotkeyService

# Initialize paths
sounds_dir = os.path.join(BASE_DIR, 'sounds')
os.makedirs(sounds_dir, exist_ok=True)

# Initialize core components
config = Config()
audio = AudioEngine(sounds_dir)
audio.set_volume(config.default_volume)

# Initialize services
hotkey_service = HotkeyService()

# Initialize eel
eel.init(WEB_DIR)


def play_sound_global(name: str):
    """Play sound from global hotkey (toggle behavior)"""
    if audio._current_playing_sound == name:
        audio.stop()
        return
    
    settings = hotkey_service.get_sound_settings(name)
    vol = settings['volume'] / 100
    
    # Apply scream mode
    if settings['scream']:
        vol = min(vol * 50.0, 50.0)
    
    # Apply pitch mode
    pitch = 1.5 if settings['pitch'] else 1.0
    
    audio.set_volume(vol)
    audio.set_pitch(pitch)
    audio.play(name)


def play_youtube_global(url: str):
    """Play YouTube from global hotkey (toggle play/pause)"""
    settings = hotkey_service.get_youtube_settings(url)
    
    vol = 50.0 if settings['scream'] else 1.0
    pitch = 1.5 if settings['pitch'] else 1.0
    
    audio.set_youtube_volume(vol)
    audio.set_youtube_pitch(pitch)
    
    info = audio.get_youtube_info()
    
    # Toggle play/pause
    if info.get('url') == url:
        if info.get('playing'):
            if info.get('paused'):
                audio.resume_youtube()
            else:
                audio.pause_youtube()
        else:
            audio.play_youtube(url)
    else:
        audio.play_youtube(url)


def stop_all_global():
    """Stop all from global hotkey"""
    audio.stop()


def update_global_hotkeys():
    """Update global hotkeys from settings"""
    hotkey_service.update_from_settings()


def on_close(page, sockets):
    """Handle browser close"""
    cleanup_and_exit()


def cleanup_and_exit():
    """Clean up all resources and exit"""
    from core.single_instance import kill_browser
    
    kill_browser()
    hotkey_service.cleanup()
    
    try:
        audio.cleanup()
    except Exception:
        pass
    
    release_lock()
    os._exit(0)


def find_browser():
    """Find available browser"""
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in chrome_paths:
        if os.path.exists(path):
            return 'chrome', path
    
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
    
    time.sleep(0.5)
    
    try:
        import psutil
        current_pid = os.getpid()
        
        for proc in psutil.process_iter(['pid', 'name', 'ppid']):
            try:
                name = proc.info['name'].lower()
                if 'chrome' in name or 'msedge' in name:
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
        pass


def main():
    """Main entry point"""
    # Handle signals
    def signal_handler(sig, frame):
        print("\n   Shutting down...")
        cleanup_and_exit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🎵 Soundboard Pro")
    print(f"   Sounds: {sounds_dir}")
    
    # Check VB-Cable
    try:
        from core.vbcable import ensure_vbcable
        if not audio.is_vb_connected():
            if ensure_vbcable():
                audio._detect_vb_cable()
    except Exception as e:
        print(f"   VB-Cable check: {e}")
    
    # Initialize API layers
    sound_api = SoundAPI(audio, sounds_dir)
    youtube_api = YouTubeAPI(audio, sounds_dir)
    settings_api = SettingsAPI(audio, update_global_hotkeys)
    
    # Initialize hotkeys
    hotkey_service.initialize(play_sound_global, play_youtube_global, stop_all_global)
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
        'port': 0,
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
        eel.start('index.html', **eel_options, block=False)
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
