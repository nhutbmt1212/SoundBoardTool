"""
Simple setup script - One command to install everything
"""
import subprocess
import sys
import os

def main():
    print("🎵 Soundboard Pro - Quick Setup")
    print("=" * 50)
    print()
    print("Installing:")
    print("  ✓ Python dependencies (pygame, pyaudio, numpy)")
    print("  ✓ VB-Audio Virtual Cable (for Discord/Game routing)")
    print()
    print("Starting installation...")
    print()
    
    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run installer
    subprocess.run([sys.executable, "installer.py"])

if __name__ == "__main__":
    main()
