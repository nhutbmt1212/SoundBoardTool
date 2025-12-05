"""
Standalone entry point with VB-Cable auto-installer
"""
import sys
import os
import tkinter as tk
from tkinter import messagebox
import zipfile
import tempfile
import subprocess
import ctypes

# Add bundled src to path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, os.path.join(bundle_dir, 'src'))
else:
    # Running as script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def is_admin():
    """Check if running with admin privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def check_vb_cable_installed():
    """Check if VB-Cable is installed"""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if 'CABLE' in info['name'].upper() or 'VB-AUDIO' in info['name'].upper():
                p.terminate()
                return True
        
        p.terminate()
        return False
    except:
        return False

def install_vb_cable():
    """Install VB-Cable from bundled installer"""
    
    if not is_admin():
        messagebox.showwarning(
            "Admin Rights Required",
            "VB-Cable installation requires Administrator privileges.\n\n"
            "Please:\n"
            "1. Close this app\n"
            "2. Right-click SoundboardPro.exe\n"
            "3. Select 'Run as Administrator'\n"
            "4. Try again"
        )
        return False
    
    response = messagebox.askyesno(
        "Install VB-Cable",
        "VB-Audio Virtual Cable is required for Discord/Game audio routing.\n\n"
        "Would you like to install it now?\n"
        "(This will take 1-2 minutes and requires a restart)"
    )
    
    if not response:
        messagebox.showinfo(
            "Optional Feature",
            "You can still use the soundboard without VB-Cable.\n\n"
            "Audio routing to Discord/Games will not be available."
        )
        return False
    
    try:
        # Extract bundled VB-Cable installer
        if getattr(sys, 'frozen', False):
            bundle_dir = sys._MEIPASS
            vb_zip = os.path.join(bundle_dir, 'vbcable_installer.zip')
        else:
            vb_zip = 'vbcable_installer.zip'
        
        if not os.path.exists(vb_zip):
            messagebox.showerror(
                "Error",
                "VB-Cable installer not found in bundle.\n\n"
                "Please download manually from:\n"
                "https://vb-audio.com/Cable/"
            )
            return False
        
        # Extract to temp directory
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(vb_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find setup executable
        setup_exe = None
        for file in os.listdir(temp_dir):
            if file.lower().endswith('.exe') and 'setup' in file.lower():
                setup_exe = os.path.join(temp_dir, file)
                break
        
        if not setup_exe:
            setup_exe = os.path.join(temp_dir, "VBCABLE_Setup_x64.exe")
        
        if not os.path.exists(setup_exe):
            messagebox.showerror(
                "Error",
                "VB-Cable setup executable not found."
            )
            return False
        
        # Show progress
        progress = tk.Toplevel()
        progress.title("Installing VB-Cable")
        progress.geometry("400x150")
        
        label = tk.Label(
            progress,
            text="Installing VB-Audio Virtual Cable...\n\nPlease wait...",
            font=("Arial", 12)
        )
        label.pack(expand=True)
        progress.update()
        
        # Run installer
        subprocess.run([setup_exe, "/passive"])
        
        progress.destroy()
        
        messagebox.showinfo(
            "Installation Complete",
            "VB-Cable installed successfully!\n\n"
            "IMPORTANT: You must restart your computer now.\n\n"
            "After restart, run SoundboardPro.exe again."
        )
        
        return True
        
    except Exception as e:
        messagebox.showerror(
            "Installation Failed",
            f"Failed to install VB-Cable:\n{str(e)}\n\n"
            "You can install manually from:\n"
            "https://vb-audio.com/Cable/"
        )
        return False

def main():
    """Main entry point for standalone executable"""
    
    # Check VB-Cable on first run
    if not check_vb_cable_installed():
        root = tk.Tk()
        root.withdraw()
        
        install_vb_cable()
        
        root.destroy()
    
    # Import and run main app
    try:
        from ui import SoundboardUI
        app = SoundboardUI()
        app.run()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Error",
            f"Failed to start Soundboard Pro:\n{str(e)}"
        )
        root.destroy()

if __name__ == "__main__":
    main()
