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
        from virtual_audio import virtual_audio
        return virtual_audio.is_installed()
    except:
        return False

def install_vb_cable():
    """Install VB-Cable from bundled installer"""
    
    if not is_admin():
        messagebox.showwarning(
            "Admin Rights Required",
            "Virtual Audio Device installation requires Administrator privileges.\n\n"
            "Please:\n"
            "1. Close this app\n"
            "2. Right-click SoundboardPro.exe\n"
            "3. Select 'Run as Administrator'\n"
            "4. Try again"
        )
        return False
    
    response = messagebox.askyesno(
        "Install Virtual Audio Device",
        "Soundboard Pro needs a Virtual Audio Device for Discord/Game routing.\n\n"
        "This will install VB-Audio Virtual Cable.\n\n"
        "Would you like to install it now?\n"
        "(Takes 1-2 minutes and requires a restart)"
    )
    
    if not response:
        messagebox.showinfo(
            "Optional Feature",
            "You can still use the soundboard without Virtual Audio Device.\n\n"
            "Audio routing to Discord/Games will not be available.\n\n"
            "You can install it later from:\n"
            "Settings → Audio Setup → Install VB-Cable"
        )
        return False
    
    try:
        from virtual_audio import virtual_audio
        
        # Get bundled installer path
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
        
        # Show progress
        progress = tk.Toplevel()
        progress.title("Installing Virtual Audio Device")
        progress.geometry("450x150")
        
        label = tk.Label(
            progress,
            text="Installing Virtual Audio Device...\n\nPlease wait...",
            font=("Arial", 12)
        )
        label.pack(expand=True)
        progress.update()
        
        # Install using virtual_audio module
        success, message = virtual_audio.install(vb_zip)
        
        progress.destroy()
        
        if success:
            # Get device names for user
            input_name = virtual_audio.get_input_device_name()
            output_name = virtual_audio.get_output_device_name()
            
            messagebox.showinfo(
                "Installation Complete",
                f"Virtual Audio Device installed successfully!\n\n"
                f"Device names:\n"
                f"  • For Soundboard: {input_name or 'CABLE Input'}\n"
                f"  • For Discord/Games: {output_name or 'CABLE Output'}\n\n"
                f"IMPORTANT: You must restart your computer now.\n\n"
                f"After restart:\n"
                f"1. Run SoundboardPro.exe\n"
                f"2. Click 'Audio Setup'\n"
                f"3. Select '{input_name or 'CABLE Input'}'\n"
                f"4. In Discord: Set Input to '{output_name or 'CABLE Output'}'"
            )
            return True
        else:
            messagebox.showerror(
                "Installation Failed",
                f"Failed to install Virtual Audio Device:\n{message}\n\n"
                "You can install manually from:\n"
                "https://vb-audio.com/Cable/"
            )
            return False
        
    except Exception as e:
        messagebox.showerror(
            "Installation Failed",
            f"Failed to install Virtual Audio Device:\n{str(e)}\n\n"
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
