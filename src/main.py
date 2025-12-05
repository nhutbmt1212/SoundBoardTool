"""
Soundboard Tool - Entry Point
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui import SoundboardUI

def main():
    app = SoundboardUI()
    app.run()

if __name__ == "__main__":
    main()
