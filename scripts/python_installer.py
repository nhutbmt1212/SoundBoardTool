"""
Python Auto-Installer
Downloads and installs Python if not present
"""
import os
import sys
import subprocess
import urllib.request
import tempfile
import platform

class PythonInstaller:
    def __init__(self):
        self.python_version = "3.11.7"
        self.temp_dir = tempfile.gettempdir()
        
        # Determine architecture
        is_64bit = platform.machine().endswith('64')
        arch = "amd64" if is_64bit else "win32"
        
        self.installer_url = f"https://www.python.org/ftp/python/{self.python_version}/python-{self.python_version}-{arch}.exe"
        self.installer_path = os.path.join(self.temp_dir, "python_installer.exe")
    
    def check_python_installed(self):
        """Check if Python is installed and get version"""
        try:
            result = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                version_str = result.stdout.strip().split()[1]
                return True, version_str
            return False, None
        except FileNotFoundError:
            return False, None
    
    def check_python_version(self, version_str):
        """Check if Python version is 3.7+"""
        try:
            major, minor, *_ = version_str.split('.')
            major, minor = int(major), int(minor)
            
            if major >= 3 and minor >= 7:
                return True
            return False
        except:
            return False
    
    def download_python(self, progress_callback=None):
        """Download Python installer"""
        print(f"📥 Downloading Python {self.python_version}...")
        print(f"URL: {self.installer_url}")
        
        def report_progress(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                progress_callback(percent)
                if percent % 10 == 0:
                    print(f"Progress: {percent}%")
        
        try:
            urllib.request.urlretrieve(
                self.installer_url,
                self.installer_path,
                reporthook=report_progress
            )
            print("✅ Download complete!")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def install_python(self):
        """Install Python"""
        print("🔧 Installing Python...")
        print("⚠️  This may take a few minutes...")
        
        try:
            # Run installer with silent flags
            # InstallAllUsers=1 - Install for all users
            # PrependPath=1 - Add to PATH
            # Include_test=0 - Don't include tests
            result = subprocess.run(
                [
                    self.installer_path,
                    "/quiet",
                    "InstallAllUsers=1",
                    "PrependPath=1",
                    "Include_test=0"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Python installed successfully!")
                return True
            else:
                print(f"⚠️  Installation may require manual confirmation")
                # Try without silent mode
                subprocess.Popen([self.installer_path])
                return True
                
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up installer file"""
        try:
            if os.path.exists(self.installer_path):
                os.remove(self.installer_path)
                print("🧹 Cleanup complete")
        except:
            pass
    
    def run_full_install(self):
        """Run complete Python installation"""
        print("=" * 60)
        print("🐍 Python Auto-Installer")
        print("=" * 60)
        print()
        
        # Check if Python is installed
        print("Step 1/3: Checking Python installation...")
        is_installed, version = self.check_python_installed()
        
        if is_installed:
            print(f"✅ Python {version} is installed")
            
            if self.check_python_version(version):
                print("✅ Python version is compatible (3.7+)")
                print()
                print("=" * 60)
                print("✅ Python is ready!")
                print("=" * 60)
                return True
            else:
                print(f"⚠️  Python {version} is too old (need 3.7+)")
                print("Please upgrade Python manually or continue with auto-install")
                print()
        else:
            print("⚠️  Python is not installed")
            print()
        
        # Ask user
        response = input("Would you like to install Python 3.11.7? (Y/n): ").strip().lower()
        if response and response != 'y':
            print("Installation cancelled.")
            return False
        
        print()
        
        # Download Python
        print("Step 2/3: Downloading Python...")
        if not self.download_python():
            print("❌ Failed to download Python")
            return False
        print()
        
        # Install Python
        print("Step 3/3: Installing Python...")
        if not self.install_python():
            print("❌ Failed to install Python")
            return False
        
        print()
        print("=" * 60)
        print("✅ Python installation complete!")
        print("=" * 60)
        print()
        print("⚠️  IMPORTANT:")
        print("1. Close this window")
        print("2. Open a NEW command prompt")
        print("3. Run: setup.bat")
        print()
        
        # Cleanup
        self.cleanup()
        
        return True

def main():
    installer = PythonInstaller()
    
    try:
        installer.run_full_install()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        installer.cleanup()
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        installer.cleanup()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
