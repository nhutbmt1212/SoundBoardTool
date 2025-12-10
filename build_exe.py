"""Build standalone executable - All in one"""
import PyInstaller.__main__
import os
import shutil
import sys
import urllib.request
import zipfile

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')
WEB = os.path.join(SRC, 'web')
SOUNDS = os.path.join(ROOT, 'sounds')
VBCABLE = os.path.join(ROOT, 'vbcable')
DIST = os.path.join(ROOT, 'dist')
BUILD = os.path.join(ROOT, 'build')

# VB-Cable download URL
VBCABLE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
VBCABLE_INSTALLER = "VBCABLE_Setup_x64.exe"


def clean():
    """Clean build folders"""
    for d in [DIST, BUILD]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print("✓ Cleaned build folders")


def download_vbcable():
    """Download VB-Cable installer if not exists"""
    installer_path = os.path.join(VBCABLE, VBCABLE_INSTALLER)
    
    if os.path.exists(installer_path):
        print("✓ VB-Cable installer already exists")
        return True
    
    print("Downloading VB-Cable installer...")
    os.makedirs(VBCABLE, exist_ok=True)
    
    try:
        zip_path = os.path.join(VBCABLE, 'vbcable.zip')
        urllib.request.urlretrieve(VBCABLE_URL, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(VBCABLE)
        
        # Clean up zip
        os.remove(zip_path)
        
        if os.path.exists(installer_path):
            print("✓ Downloaded VB-Cable installer")
            return True
        else:
            print("⚠ VB-Cable installer not found in zip")
            return False
    except Exception as e:
        print(f"⚠ Failed to download VB-Cable: {e}")
        return False


def build():
    """Build executable"""
    print("Building SoundboardPro.exe...")
    
    args = [
        os.path.join(SRC, 'app.py'),
        '--name=SoundboardPro',
        '--onefile',
        '--windowed',
        '--noconfirm',
        '--clean',
        f'--icon={os.path.join(WEB, "assets", "icon.ico")}',
        # Add web folder
        f'--add-data={WEB}{os.pathsep}web',
        # Add VB-Cable installer
        f'--add-data={VBCABLE}{os.pathsep}vbcable',
        # Hidden imports - Core
        '--hidden-import=eel',
        '--hidden-import=bottle',
        '--hidden-import=gevent',
        '--hidden-import=geventwebsocket',
        '--hidden-import=gevent.ssl',
        '--hidden-import=gevent._ssl3',
        # Hidden imports - Audio
        '--hidden-import=pygame',
        '--hidden-import=pygame.mixer',
        '--hidden-import=pygame.sndarray',
        '--hidden-import=pygame.base',
        '--hidden-import=sounddevice',
        '--hidden-import=scipy',
        '--hidden-import=scipy.io',
        '--hidden-import=scipy.io.wavfile',
        '--hidden-import=scipy.io.wavfile',
        '--hidden-import=numpy',
        '--hidden-import=pyrnnoise',
        # Hidden imports - Hotkeys (optional)
        '--hidden-import=keyboard',
        # Hidden imports - Process management
        '--hidden-import=psutil',
        # Hidden imports - GUI
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.filedialog',
        # Collect all
        '--collect-all=eel',
        '--collect-all=pygame',
        '--collect-all=sounddevice',
        '--collect-all=sounddevice',
        '--collect-all=yt_dlp',
        '--collect-all=pyrnnoise',
        # Exclude unnecessary
        '--exclude-module=matplotlib',
        '--exclude-module=PIL',
        '--exclude-module=cv2',
        '--exclude-module=tkinter.test',
        '--exclude-module=pytest',
    ]
    
    PyInstaller.__main__.run(args)
    print("✓ Built executable")


def copy_assets():
    """Copy sounds folder to dist"""
    dist_sounds = os.path.join(DIST, 'sounds')
    os.makedirs(dist_sounds, exist_ok=True)
    
    # Copy .gitkeep if exists
    gitkeep = os.path.join(SOUNDS, '.gitkeep')
    if os.path.exists(gitkeep):
        shutil.copy(gitkeep, dist_sounds)
    
    print("✓ Created sounds folder")


def main():
    print("=" * 50)
    print("  Soundboard Pro - Build")
    print("=" * 50)
    
    clean()
    download_vbcable()
    build()
    copy_assets()
    
    exe_path = os.path.join(DIST, 'SoundboardPro.exe')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print("=" * 50)
        print(f"✅ Build complete!")
        print(f"   Output: {exe_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print("=" * 50)
    else:
        print("❌ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
