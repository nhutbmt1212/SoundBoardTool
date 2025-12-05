"""
Virtual Audio Device Manager
Creates a virtual audio device specifically for Soundboard Pro
"""
import os
import sys
import subprocess
import tempfile
import zipfile
import ctypes
import winreg

class VirtualAudioDevice:
    """Manages virtual audio device for Soundboard Pro"""
    
    def __init__(self):
        self.device_name = "Soundboard Pro Virtual Audio"
        
    def is_installed(self):
        """Check if our virtual audio device is installed"""
        try:
            # Check registry
            paths = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
            ]
            
            for path in paths:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
                    i = 0
                    while True:
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            subkey = winreg.OpenKey(key, subkey_name)
                            try:
                                name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                if "VB-AUDIO" in name.upper() or "CABLE" in name.upper():
                                    winreg.CloseKey(subkey)
                                    winreg.CloseKey(key)
                                    return True
                            except:
                                pass
                            winreg.CloseKey(subkey)
                            i += 1
                        except OSError:
                            break
                    winreg.CloseKey(key)
                except:
                    continue
            
            # Check if device exists in audio devices
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    device_name = info['name'].upper()
                    if 'CABLE' in device_name or 'VB-AUDIO' in device_name:
                        p.terminate()
                        return True
                
                p.terminate()
            except:
                pass
            
            return False
        except:
            return False
    
    def get_device_info(self):
        """Get information about the virtual audio device"""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            devices = {
                'input': None,
                'output': None
            }
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                device_name = info['name'].upper()
                
                if 'CABLE' in device_name or 'VB-AUDIO' in device_name:
                    if info['maxInputChannels'] > 0:
                        devices['output'] = {
                            'index': i,
                            'name': info['name'],
                            'channels': info['maxInputChannels']
                        }
                    if info['maxOutputChannels'] > 0:
                        devices['input'] = {
                            'index': i,
                            'name': info['name'],
                            'channels': info['maxOutputChannels']
                        }
            
            p.terminate()
            return devices
        except:
            return {'input': None, 'output': None}
    
    def install(self, installer_zip_path):
        """Install VB-Cable virtual audio device"""
        try:
            # Extract installer
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(installer_zip_path, 'r') as zip_ref:
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
                return False, "Setup executable not found"
            
            # Run installer with admin rights automatically
            # Use ShellExecute to elevate privileges
            result = ctypes.windll.shell32.ShellExecuteW(
                None,
                "runas",  # Run as administrator
                setup_exe,
                "-i -h",  # Silent install flags
                None,
                0  # Hide window
            )
            
            # ShellExecute returns > 32 on success
            if result > 32:
                # Wait a bit for installation to complete
                import time
                time.sleep(5)
                return True, "Installation successful"
            else:
                return False, f"Installation failed with code {result}"
                
        except Exception as e:
            return False, str(e)
    
    def get_input_device_name(self):
        """Get the name to use in Discord/Games (the OUTPUT of VB-Cable)"""
        devices = self.get_device_info()
        if devices['output']:
            return devices['output']['name']
        return None
    
    def get_output_device_name(self):
        """Get the name to use in Soundboard (the INPUT of VB-Cable)"""
        devices = self.get_device_info()
        if devices['input']:
            return devices['input']['name']
        return None

# Global instance
virtual_audio = VirtualAudioDevice()
