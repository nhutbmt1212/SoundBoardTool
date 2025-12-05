"""
Build script to create TRUE standalone executable
No Python, no dependencies needed - everything bundled!
"""
import PyInstaller.__main__
import os
import shutil
import urllib.request

def download_vb_cable():
    """Download VB-Cable installer to bundle"""
    print("Downloading VB-Cable installer to bundle...")
    
    url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
    output = "vbcable_installer.zip"
    
    if not os.path.exists(output):
        try:
            urllib.request.urlretrieve(url, output)
            print(f"✅ Downloaded: {output}")
        except Exception as e:
            print(f"⚠️  Could not download VB-Cable: {e}")
            print("   Continuing without VB-Cable bundled...")
    else:
        print(f"✅ Already exists: {output}")

def build():
    """Build TRUE standalone executable"""
    
    print("=" * 60)
    print("Building TRUE Standalone Soundboard Pro")
    print("No Python or dependencies needed!")
    print("=" * 60)
    print()
    
    # Download VB-Cable to bundle
    download_vb_cable()
    
    # Clean previous builds
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    
    print("\nBuilding executable...")
    print("This will take 2-3 minutes...")
    print()
    
    # Determine if VB-Cable is available
    vb_cable_args = []
    if os.path.exists('vbcable_installer.zip'):
        vb_cable_args = ['--add-data=vbcable_installer.zip;.']
        print("✅ VB-Cable will be bundled")
    else:
        print("⚠️  VB-Cable not bundled (will need manual install)")
    
    # PyInstaller arguments
    args = [
        'src/main_standalone.py',           # Standalone entry point
        '--name=SoundboardPro',             # Executable name
        '--onefile',                        # Single file
        '--windowed',                       # No console (GUI only)
        '--noconfirm',                      # Overwrite without asking
        
        # Add all source files
        '--add-data=src;src',
        '--add-data=sounds;sounds',
        
        # Hidden imports - all dependencies
        '--hidden-import=pygame',
        '--hidden-import=pygame.mixer',
        '--hidden-import=pygame_ce',
        '--hidden-import=pyaudio',
        '--hidden-import=numpy',
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.filedialog',
        '--hidden-import=tkinter.messagebox',
        
        # Collect all pygame files
        '--collect-all=pygame',
        '--collect-all=pygame_ce',
        
        # Exclude unnecessary modules to reduce size
        '--exclude-module=matplotlib',
        '--exclude-module=scipy',
        '--exclude-module=pandas',
        '--exclude-module=PIL',
        '--exclude-module=IPython',
        '--exclude-module=notebook',
    ]
    
    # Add VB-Cable if available
    args.extend(vb_cable_args)
    
    PyInstaller.__main__.run(args)
    
    print("\n" + "=" * 60)
    print("✅ Build Complete!")
    print("=" * 60)
    print()
    print(f"Executable: dist/SoundboardPro.exe")
    print()
    print("Features:")
    print("  ✅ No Python installation needed")
    print("  ✅ All libraries bundled (pygame, pyaudio, numpy)")
    print("  ✅ VB-Cable installer included (if downloaded)")
    print("  ✅ True standalone - works offline")
    print("  ✅ Single EXE file")
    print()
    print("User Experience:")
    print("  1. Double-click SoundboardPro.exe")
    print("  2. First run: Optionally install VB-Cable")
    print("  3. App starts immediately - no setup needed!")
    print()
    
    # Get file size
    exe_path = "dist/SoundboardPro.exe"
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")
        print()

if __name__ == "__main__":
    build()
