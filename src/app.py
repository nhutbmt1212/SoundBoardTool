"""Soundboard Pro - Web UI Application (Refactored)"""
import sys
import os
import signal
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

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
from core.config_paths import (
    get_sounds_dir, 
    get_config_path, 
    migrate_old_configs,
    get_vbcable_installer_path
)
from api import SoundAPI, YouTubeAPI, SettingsAPI, TikTokAPI
from api.tts_api import TTSAPI
from api import backup_api  # Import to expose backup endpoints
from services import HotkeyService

# Migrate old configs to AppData
migrate_old_configs()

# Initialize paths (now using AppData)
sounds_dir = get_sounds_dir()

# Initialize core components
config = Config()
audio = AudioEngine(sounds_dir)
audio.set_volume(config.default_volume)

# Initialize services
hotkey_service = HotkeyService()

# Initialize eel
eel.init(WEB_DIR)

# Constants
MAX_SCREAM_MULTIPLIER = 50.0
PITCH_MODE_MULTIPLIER = 1.5
DEBOUNCE_SECONDS = 0.5


def _apply_playback_settings(settings, all_settings, trim_key, identifier):
    """Helper to apply common playback settings (volume, pitch, trim)
    
    Args:
        settings: Individual item settings (volume, scream, pitch)
        all_settings: All application settings
        trim_key: Key for trim settings ('trimSettings', 'youtubeTrimSettings', etc.)
        identifier: Sound name or URL for trim lookup
    
    Returns:
        tuple: (volume, pitch, trim_start, trim_end)
    """
    vol = settings['volume'] / 100
    
    # Apply scream mode
    if settings['scream']:
        vol = min(vol * MAX_SCREAM_MULTIPLIER, MAX_SCREAM_MULTIPLIER)
    
    # Apply pitch mode
    pitch = PITCH_MODE_MULTIPLIER if settings['pitch'] else 1.0
    
    # Load trim settings
    trim_settings = all_settings.get(trim_key, {})
    trim = trim_settings.get(identifier, {})
    trim_start = trim.get('start', 0)
    trim_end = trim.get('end', 0)
    
    return vol, pitch, trim_start, trim_end


def play_sound_global(name: str):
    """Play sound from global hotkey (toggle behavior)"""
    from core.config import load_sound_settings
    
    if audio._current_playing_sound == name:
        audio.stop()
        return
    
    settings = hotkey_service.get_sound_settings(name)
    all_settings = load_sound_settings()
    
    vol, pitch, trim_start, trim_end = _apply_playback_settings(
        settings, all_settings, 'trimSettings', name
    )
    
    audio.set_volume(vol)
    audio.set_pitch(pitch)
    audio.set_trim(trim_start, trim_end)
    
    # Apply effects
    effects = settings.get('effects', {})
    audio.set_sound_effects(effects)
    
    audio.play(name)


def play_youtube_global(url: str):
    """Play YouTube from global hotkey (toggle play/stop)"""
    from core.config import load_sound_settings
    
    settings = hotkey_service.get_youtube_settings(url)
    all_settings = load_sound_settings()
    
    vol, pitch, trim_start, trim_end = _apply_playback_settings(
        settings, all_settings, 'youtubeTrimSettings', url
    )
    
    audio.set_youtube_volume(vol)
    audio.set_youtube_pitch(pitch)
    audio.set_youtube_trim(trim_start, trim_end)
    
    # Apply effects
    effects = settings.get('effects', {})
    audio.set_youtube_effects(effects)
    
    info = audio.get_youtube_info()
    
    # Toggle play/stop (not pause)
    if info.get('url') == url and info.get('playing'):
        audio.stop_youtube()
    else:
        loop = settings.get('loop', False)
        audio.play_youtube(url, loop=loop)


def play_tiktok_global(url: str):
    """Play TikTok from global hotkey (toggle play/stop)"""
    from core.config import load_sound_settings
    
    settings = hotkey_service.get_tiktok_settings(url)
    all_settings = load_sound_settings()
    
    vol, pitch, trim_start, trim_end = _apply_playback_settings(
        settings, all_settings, 'tiktokTrimSettings', url
    )
    
    audio.set_tiktok_volume(vol)
    audio.set_tiktok_pitch(pitch)
    audio.set_tiktok_trim(trim_start, trim_end)
    
    # Apply effects
    effects = settings.get('effects', {})
    audio.set_tiktok_effects(effects)
    
    info = audio.get_tiktok_info()
    
    # Toggle play/stop (not pause)
    if info.get('url') == url and info.get('playing'):
        audio.stop_tiktok()
    else:
        loop = settings.get('loop', False)
        audio.play_tiktok(url, loop=loop)


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
    
    # Release lock first so we can restart even if cleanup hangs
    try:
        release_lock()
    except Exception as e:
        logger.debug(f"Failed to release lock: {e}")

    try:
        kill_browser()
    except Exception as e:
        logger.debug(f"Failed to kill browser: {e}")

    try:
        hotkey_service.cleanup()
    except Exception as e:
        logger.debug(f"Failed to cleanup hotkey service: {e}")
    
    try:
        audio.cleanup()
    except Exception as e:
        logger.debug(f"Failed to cleanup audio: {e}")
    
    # Force exit
    os._exit(0)


def show_vbcable_install_dialog():
    """Show VB-Cable installation dialog and wait for install"""
    try:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        installer_path = get_vbcable_installer_path()
        
        if installer_path:
            result = messagebox.askyesno(
                "VB-Cable Required",
                "VB-Cable is required to run Soundboard Pro.\n\n"
                "Would you like to install it now?\n\n"
                "(Requires administrator rights)",
                icon='warning'
            )
            
            if result:
                # Run installer
                try:
                    subprocess.run([installer_path], check=False)
                    messagebox.showinfo(
                        "Installation Complete",
                        "Please restart Soundboard Pro after VB-Cable installation."
                    )
                except Exception as e:
                    messagebox.showerror(
                        "Installation Failed",
                        f"Failed to run installer: {e}\n\n"
                        "Please install VB-Cable manually from:\n"
                        "https://vb-audio.com/Cable/"
                    )
            else:
                messagebox.showinfo(
                    "Installation Required",
                    "VB-Cable is required to run Soundboard Pro.\n\n"
                    "Download from: https://vb-audio.com/Cable/"
                )
        else:
            messagebox.showerror(
                "VB-Cable Required",
                "VB-Cable is required to run Soundboard Pro.\n\n"
                "Download from: https://vb-audio.com/Cable/"
            )
        
        root.destroy()
        return False  # Exit app
        
    except ImportError:
        # Tkinter not available, print to console
        print("\n" + "="*50)
        print("  ⚠️  VB-Cable Required")
        print("="*50)
        print("VB-Cable is required to run Soundboard Pro.")
        print("\nDownload from: https://vb-audio.com/Cable/")
        print("\nAfter installation, restart Soundboard Pro.")
        print("="*50 + "\n")
        return False


def find_browser():
    """Find available browser with fallback chain"""
    # Try Edge first (most common on Windows)
    edge_paths = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in edge_paths:
        if os.path.exists(path):
            return 'edge', path
    
    # Try Chrome
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in chrome_paths:
        if os.path.exists(path):
            return 'chrome', path
    
    # Try Firefox
    firefox_paths = [
        os.path.expandvars(r"%ProgramFiles%\Mozilla Firefox\firefox.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe"),
    ]
    for path in firefox_paths:
        if os.path.exists(path):
            return 'firefox', path
    
    # Fallback to default
    return 'default', None


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
    
    print("🎵 Dalit")
    print(f"   Sounds: {sounds_dir}")
    
    # Check VB-Cable (REQUIRED)
    if not audio.is_vb_connected():
        print("   VB-Cable: not detected")
        if not show_vbcable_install_dialog():
            print("   Exiting...")
            sys.exit(1)
    else:
        print("   VB-Cable: connected")
    
    # Initialize API layers
    sound_api = SoundAPI(audio, sounds_dir)
    youtube_api = YouTubeAPI(audio, sounds_dir)
    tiktok_api = TikTokAPI(audio, sounds_dir)
    settings_api = SettingsAPI(audio, update_global_hotkeys)
    tts_api = TTSAPI(audio)
    
    # Initialize hotkeys
    hotkey_service.initialize(play_sound_global, play_youtube_global, play_tiktok_global, stop_all_global)
    try:
        update_global_hotkeys()
        print("   Hotkeys: loaded")
    except Exception as e:
        print(f"   Hotkeys: failed ({e})")
    
    # Find browser with fallback chain
    browser_mode, browser_path = find_browser()
    print(f"   Browser: {browser_mode}")
    
    eel_options = {
        'size': (1100, 750),
        'close_callback': on_close,
        'port': 8000,  # App runs on 8000, OAuth callback uses 8080
        'cmdline_args': ['--disable-extensions']
    }
    
    print("   Starting...")
    
    # Try browsers in order: detected → chrome → edge → firefox → default
    browsers_to_try = [browser_mode]
    if browser_mode != 'chrome':
        browsers_to_try.append('chrome')
    if browser_mode != 'edge':
        browsers_to_try.append('edge')
    if browser_mode != 'firefox':
        browsers_to_try.append('firefox')
    if 'default' not in browsers_to_try:
        browsers_to_try.append('default')
    
    started = False
    for browser in browsers_to_try:
        try:
            eel_options['mode'] = browser
            eel.start('index.html', **eel_options, block=False)
            track_browser_process()
            print(f"   ✓ Started with {browser}")
            started = True
            break
        except Exception as e:
            if browser == browsers_to_try[-1]:
                # Last browser failed
                print(f"   ✗ Failed to start with any browser: {e}")
                raise
            else:
                # Try next browser
                continue
    
    if started:
        try:
            # Keep main thread alive
            while True:
                eel.sleep(1.0)
        except SystemExit:
            pass
        except KeyboardInterrupt:
            print("\n   Interrupted")
    
    cleanup_and_exit()


if __name__ == "__main__":
    main()
