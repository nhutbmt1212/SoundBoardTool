"""Build standalone executable - All in one"""
import PyInstaller.__main__
import os
import shutil
import sys

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, 'src')
WEB = os.path.join(SRC, 'web')
SOUNDS = os.path.join(ROOT, 'sounds')
DIST = os.path.join(ROOT, 'dist')
BUILD = os.path.join(ROOT, 'build')

def clean():
    """Clean build folders"""
    for d in [DIST, BUILD]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print("✓ Cleaned build folders")

def build():
    """Build executable"""
    print("Building SoundboardPro.exe...")
    
    # PyInstaller args
    args = [
        os.path.join(SRC, 'app.py'),
        '--name=SoundboardPro',
        '--onefile',
        '--windowed',
        '--noconfirm',
        '--clean',
        # Add web folder
        f'--add-data={WEB}{os.pathsep}web',
        # Hidden imports
        '--hidden-import=pygame',
        '--hidden-import=pygame.mixer',
        '--hidden-import=pygame.sndarray',
        '--hidden-import=sounddevice',
        '--hidden-import=scipy',
        '--hidden-import=scipy.io',
        '--hidden-import=scipy.io.wavfile',
        '--hidden-import=numpy',
        '--hidden-import=eel',
        '--hidden-import=bottle',
        '--hidden-import=gevent',
        '--hidden-import=geventwebsocket',
        # Collect all for eel
        '--collect-all=eel',
        # Exclude unnecessary
        '--exclude-module=tkinter.test',
        '--exclude-module=unittest',
        '--exclude-module=pytest',
    ]
    
    PyInstaller.__main__.run(args)
    print("✓ Built executable")

def copy_assets():
    """Copy sounds folder to dist"""
    dist_sounds = os.path.join(DIST, 'sounds')
    
    if os.path.exists(SOUNDS):
        # Copy only .gitkeep, not actual sounds
        os.makedirs(dist_sounds, exist_ok=True)
        gitkeep = os.path.join(SOUNDS, '.gitkeep')
        if os.path.exists(gitkeep):
            shutil.copy(gitkeep, dist_sounds)
    else:
        os.makedirs(dist_sounds, exist_ok=True)
    
    print("✓ Created sounds folder")

def main():
    print("=" * 50)
    print("  Soundboard Pro - Build")
    print("=" * 50)
    
    clean()
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
