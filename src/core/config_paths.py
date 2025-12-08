"""Centralized configuration paths management - Use AppData for user data"""
import os
import sys
import shutil

def get_app_data_dir():
    """Get AppData directory for SoundboardPro"""
    appdata = os.getenv('APPDATA')
    if not appdata:
        # Fallback for non-Windows or missing APPDATA
        appdata = os.path.expanduser('~')
    
    app_dir = os.path.join(appdata, 'SoundboardPro')
    os.makedirs(app_dir, exist_ok=True)
    return app_dir


def get_config_path(filename):
    """Get path for config file in AppData"""
    return os.path.join(get_app_data_dir(), filename)


def get_sounds_dir():
    """Get sounds directory in AppData"""
    sounds_dir = os.path.join(get_app_data_dir(), 'sounds')
    os.makedirs(sounds_dir, exist_ok=True)
    return sounds_dir


def get_youtube_cache_dir():
    """Get YouTube cache directory in AppData"""
    cache_dir = os.path.join(get_app_data_dir(), 'youtube_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_tiktok_cache_dir():
    """Get TikTok cache directory in AppData"""
    cache_dir = os.path.join(get_app_data_dir(), 'tiktok_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def migrate_old_configs():
    """Migrate old configs from exe directory to AppData"""
    # Get exe directory (where old configs might be)
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        exe_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        exe_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Files to migrate
    old_files = [
        'sound_settings.json',
        'soundboard_config.json'
    ]
    
    migrated = []
    for filename in old_files:
        old_path = os.path.join(exe_dir, filename)
        new_path = get_config_path(filename)
        
        # Only migrate if old exists and new doesn't
        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                shutil.copy(old_path, new_path)
                migrated.append(filename)
                print(f"✓ Migrated {filename} to AppData")
            except Exception as e:
                print(f"⚠ Failed to migrate {filename}: {e}")
    
    # Migrate sounds directory
    old_sounds = os.path.join(exe_dir, 'sounds')
    new_sounds = get_sounds_dir()
    
    if os.path.exists(old_sounds) and os.path.isdir(old_sounds):
        # Copy all sound files
        try:
            for filename in os.listdir(old_sounds):
                if filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    old_file = os.path.join(old_sounds, filename)
                    new_file = os.path.join(new_sounds, filename)
                    if not os.path.exists(new_file):
                        shutil.copy(old_file, new_file)
                        migrated.append(f"sounds/{filename}")
            if migrated:
                print(f"✓ Migrated {len([f for f in migrated if f.startswith('sounds/')])} sound files")
        except Exception as e:
            print(f"⚠ Failed to migrate sounds: {e}")
    
    return migrated


def get_vbcable_installer_path():
    """Get path to bundled VB-Cable installer"""
    if getattr(sys, 'frozen', False):
        # Running as exe - installer is bundled
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    installer_path = os.path.join(base_path, 'vbcable', 'VBCABLE_Setup_x64.exe')
    return installer_path if os.path.exists(installer_path) else None
