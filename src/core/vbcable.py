"""VB-Cable Auto Installer"""
import os
import sys
import subprocess
import tempfile
import zipfile
import urllib.request

VBCABLE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
VBCABLE_INSTALLER = "VBCABLE_Setup_x64.exe"


def is_vbcable_installed() -> bool:
    """Check if VB-Cable is installed via registry and device detection"""
    # Method 1: Check Windows Registry
    try:
        import winreg
        # Check if VB-Cable driver is registered
        key_paths = [
            r"SOFTWARE\VB-Audio\VBCABLE",
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\VB-Audio Virtual Cable",
        ]
        
        for key_path in key_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, winreg.KEY_READ):
                    return True
            except FileNotFoundError:
                continue
    except Exception:
        pass
    
    # Method 2: Check audio devices
    try:
        import sounddevice as sd
        for dev in sd.query_devices():
            name_lower = dev['name'].lower()
            if 'vb-audio virtual cable' in name_lower or 'cable input' in name_lower or 'cable output' in name_lower:
                return True
    except Exception:
        pass
    
    return False


def get_bundled_installer() -> str | None:
    """Get path to bundled VB-Cable installer"""
    if getattr(sys, 'frozen', False):
        # PyInstaller bundle
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    installer = os.path.join(base, 'vbcable', VBCABLE_INSTALLER)
    if os.path.exists(installer):
        return installer
    return None


def download_vbcable() -> str | None:
    """Download VB-Cable installer to temp folder"""
    try:
        temp_dir = tempfile.mkdtemp(prefix='vbcable_')
        zip_path = os.path.join(temp_dir, 'vbcable.zip')
        
        print("   Downloading VB-Cable...")
        urllib.request.urlretrieve(VBCABLE_URL, zip_path)
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)
        
        installer = os.path.join(temp_dir, VBCABLE_INSTALLER)
        if os.path.exists(installer):
            return installer
        
        # Try finding it
        for f in os.listdir(temp_dir):
            if f.lower().endswith('.exe') and 'x64' in f.lower():
                return os.path.join(temp_dir, f)
    except Exception as e:
        print(f"   Download failed: {e}")
    return None


def install_vbcable(silent: bool = True) -> bool:
    """Install VB-Cable driver"""
    if is_vbcable_installed():
        print("   VB-Cable already installed")
        return True
    
    # Try bundled first
    installer = get_bundled_installer()
    
    # Download if not bundled
    if not installer:
        installer = download_vbcable()
    
    if not installer:
        print("   Could not get VB-Cable installer")
        return False
    
    try:
        print("   Installing VB-Cable (requires admin)...")
        
        # Run installer with admin rights
        if silent:
            # Silent install - requires admin
            result = subprocess.run(
                ['powershell', '-Command', 
                 f'Start-Process -FilePath "{installer}" -ArgumentList "/S" -Verb RunAs -Wait'],
                capture_output=True,
                timeout=120
            )
        else:
            # Interactive install
            result = subprocess.run(
                ['powershell', '-Command',
                 f'Start-Process -FilePath "{installer}" -Verb RunAs -Wait'],
                timeout=120
            )
        
        # Check if installed
        import time
        time.sleep(3)  # Wait longer for driver to register
        
        if is_vbcable_installed():
            print("   ✓ VB-Cable installed!")
            return True
        else:
            print("   ⚠ VB-Cable installed but may require system restart")
            print("   Please restart your computer to complete installation")
            return False
            
    except subprocess.TimeoutExpired:
        print("   Install timed out")
    except Exception as e:
        print(f"   Install error: {e}")
    
    return False


def ensure_vbcable() -> bool:
    """Ensure VB-Cable is installed, install if not"""
    # Check if already installed
    if is_vbcable_installed():
        print("✓ VB-Cable is installed")
        return True
    
    print("⚠ VB-Cable not found")
    
    # Check if user previously declined installation
    try:
        from core.config_paths import get_config_dir
        skip_file = os.path.join(get_config_dir(), '.skip_vbcable')
        if os.path.exists(skip_file):
            print("   (Installation skipped by user)")
            return False
    except Exception:
        pass
    
    # Ask user
    try:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        result = messagebox.askyesnocancel(
            "VB-Cable Required",
            "VB-Cable virtual audio driver is not installed.\n\n"
            "This is required for routing audio to Discord/OBS.\n\n"
            "Install VB-Cable now?\n\n"
            "Note: Installation requires admin rights and may need a restart.",
            parent=root
        )
        root.destroy()
        
        if result is True:
            # User clicked Yes
            success = install_vbcable(silent=False)
            if not success:
                messagebox.showinfo(
                    "Restart Required",
                    "VB-Cable installation complete.\n\n"
                    "Please restart your computer to finish setup.",
                    parent=None
                )
            return success
        elif result is False:
            # User clicked No - don't ask again
            try:
                from core.config_paths import get_config_dir
                skip_file = os.path.join(get_config_dir(), '.skip_vbcable')
                with open(skip_file, 'w') as f:
                    f.write('1')
            except Exception:
                pass
            return False
        else:
            # User clicked Cancel - ask again next time
            return False
            
    except Exception:
        # No GUI, try silent install
        return install_vbcable(silent=True)
    
    return False
