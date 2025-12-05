"""
Auto Installer for Soundboard Pro
Automatically downloads and installs VB-Audio Virtual Cable
"""
import os
import sys
import subprocess
import urllib.request
import zipfile
import tempfile
import ctypes
from pathlib import Path

class SoundboardInstaller:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.vb_cable_url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
        self.vb_cable_zip = os.path.join(self.temp_dir, "vbcable.zip")
        self.vb_cable_dir = os.path.join(self.temp_dir, "vbcable")
        
    def is_admin(self):
        """Check if running with admin privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def run_as_admin(self):
        """Restart script with admin privileges"""
        if sys.platform == 'win32':
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
    
    def download_vb_cable(self, progress_callback=None):
        """Download VB-Audio Virtual Cable"""
        print("📥 Downloading VB-Audio Virtual Cable...")
        
        def report_progress(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                progress_callback(percent)
        
        try:
            urllib.request.urlretrieve(
                self.vb_cable_url,
                self.vb_cable_zip,
                reporthook=report_progress
            )
            print("✅ Download complete!")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def extract_vb_cable(self):
        """Extract VB-Cable zip file"""
        print("📦 Extracting files...")
        try:
            os.makedirs(self.vb_cable_dir, exist_ok=True)
            with zipfile.ZipFile(self.vb_cable_zip, 'r') as zip_ref:
                zip_ref.extractall(self.vb_cable_dir)
            print("✅ Extraction complete!")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def install_vb_cable(self):
        """Install VB-Cable driver with multiple methods"""
        print("🔧 Installing VB-Audio Virtual Cable...")
        
        try:
            # Find the setup executable
            setup_exe = None
            for file in os.listdir(self.vb_cable_dir):
                if file.lower().endswith('.exe') and 'setup' in file.lower():
                    setup_exe = os.path.join(self.vb_cable_dir, file)
                    break
            
            if not setup_exe:
                # Try x64 version
                setup_exe = os.path.join(self.vb_cable_dir, "VBCABLE_Setup_x64.exe")
                if not os.path.exists(setup_exe):
                    setup_exe = os.path.join(self.vb_cable_dir, "VBCABLE_Setup.exe")
            
            if not os.path.exists(setup_exe):
                print(f"❌ Setup executable not found in {self.vb_cable_dir}")
                return False
            
            print(f"📌 Found installer: {setup_exe}")
            
            # Method 1: Try PowerShell with admin elevation (most reliable)
            print("🔄 Method 1: Using PowerShell with admin elevation...")
            try:
                ps_command = f'Start-Process -FilePath "{setup_exe}" -ArgumentList "-i","-h" -Verb RunAs -Wait'
                result = subprocess.run(
                    ["powershell", "-Command", ps_command],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    print("✅ PowerShell installation completed!")
                    import time
                    time.sleep(5)
                    
                    if self.check_vb_cable_installed():
                        print("✅ VB-Cable installed and verified!")
                        return True
                    else:
                        print("⚠️  Installation completed - restart required")
                        return True
            except subprocess.TimeoutExpired:
                print("⏳ Installation taking longer than expected...")
                print("⚠️  This is normal - continuing...")
                return True
            except Exception as e:
                print(f"⚠️  PowerShell method failed: {e}")
            
            # Method 2: Try direct ShellExecute
            print("🔄 Method 2: Using ShellExecute...")
            try:
                result = ctypes.windll.shell32.ShellExecuteW(
                    None,
                    "runas",
                    setup_exe,
                    "-i -h",
                    None,
                    1  # Show window
                )
                
                if result > 32:
                    print("✅ Installation started!")
                    print("⏳ Waiting for installation...")
                    import time
                    time.sleep(15)
                    print("⚠️  Installation completed - restart required")
                    return True
                else:
                    print(f"⚠️  ShellExecute returned: {result}")
            except Exception as e:
                print(f"⚠️  ShellExecute method failed: {e}")
            
            # Method 3: Try subprocess with runas
            print("🔄 Method 3: Using subprocess with runas...")
            try:
                result = subprocess.run(
                    ["runas", "/user:Administrator", f'"{setup_exe}" -i -h'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    print("✅ Installation completed!")
                    return True
            except Exception as e:
                print(f"⚠️  Runas method failed: {e}")
            
            # Method 4: Try without silent flags (user will see installer)
            print("🔄 Method 4: Running installer with UI...")
            try:
                ps_command = f'Start-Process -FilePath "{setup_exe}" -Verb RunAs -Wait'
                result = subprocess.run(
                    ["powershell", "-Command", ps_command],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                print("✅ Installation completed!")
                return True
            except subprocess.TimeoutExpired:
                print("⚠️  Installation may still be running")
                print("    Please wait for installer to finish")
                return True
            except Exception as e:
                print(f"⚠️  UI method failed: {e}")
            
            print("❌ All installation methods failed")
            print("📝 Please install manually:")
            print(f"   1. Go to: {self.vb_cable_dir}")
            print(f"   2. Right-click: {os.path.basename(setup_exe)}")
            print("   3. Select: Run as Administrator")
            print("   4. Click: Install Driver")
            return False
                
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_vb_cable_installed(self):
        """Check if VB-Cable is already installed"""
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
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing Python dependencies...")
        try:
            # Get requirements.txt path (in parent directory)
            req_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", "-r", req_file
            ])
            print("✅ Dependencies installed!")
            return True
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print(f"    Try manually: pip install -r requirements.txt")
            return False
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.vb_cable_zip):
                os.remove(self.vb_cable_zip)
            if os.path.exists(self.vb_cable_dir):
                import shutil
                shutil.rmtree(self.vb_cable_dir)
        except:
            pass
    
    def run_full_install(self):
        """Run complete installation process"""
        print("=" * 60)
        print("🎵 Soundboard Pro - Auto Installer")
        print("=" * 60)
        print()
        
        # Step 1: Install Python dependencies
        print("[1/4] Installing Python dependencies...")
        if not self.install_python_dependencies():
            print("⚠️  Warning: Some dependencies may not be installed")
            print("    You can install manually later with: pip install -r requirements.txt")
        print()
        
        # Step 2: Check if VB-Cable already installed
        print("[2/4] Checking for VB-Audio Virtual Cable...")
        if self.check_vb_cable_installed():
            print("✅ VB-Cable is already installed!")
            print()
            print("=" * 60)
            print("✅ Setup complete!")
            print("=" * 60)
            print()
            print("Next steps:")
            print("  1. Restart your computer")
            print("  2. Run: run.bat")
            print()
            return True
        
        print("⚠️  VB-Cable not found. Installing...")
        print()
        
        # Step 3: Download VB-Cable
        print("[3/4] Downloading VB-Audio Virtual Cable...")
        if not self.download_vb_cable():
            print("❌ Download failed")
            print("    You can download manually from: https://vb-audio.com/Cable/")
            return False
        print()
        
        # Step 4: Extract and Install
        print("[4/4] Installing VB-Audio Virtual Cable...")
        if not self.extract_vb_cable():
            print("❌ Extraction failed")
            return False
        
        if not self.install_vb_cable():
            print("❌ Installation failed")
            print("    You may need to run as Administrator")
            return False
        
        print()
        print("=" * 60)
        print("✅ Installation complete!")
        print("=" * 60)
        print()
        print("⚠️  IMPORTANT: Restart your computer now")
        print("After restart, run: run.bat")
        print()
        
        # Cleanup
        self.cleanup()
        
        return True

def main():
    installer = SoundboardInstaller()
    
    try:
        installer.run_full_install()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        installer.cleanup()
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print(f"    Error details: {str(e)}")
        installer.cleanup()
    
    print("\nSetup finished. Press Enter to exit...")
    try:
        input()
    except:
        pass

if __name__ == "__main__":
    main()
