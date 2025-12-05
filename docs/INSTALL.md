# 🚀 Installation Guide - Zero Manual Setup

## ⚡ Quick Install (Recommended)

### Windows

**Option 1: Double-click setup (Easiest)**
```
1. Double-click setup.bat
2. Wait for installation to complete
3. Restart computer
4. Run: python main.py
```

**Option 2: Command line**
```bash
python setup.py
```

### What gets installed automatically?

✅ **Python Dependencies**
- pygame-ce (audio playback)
- pyaudio (audio routing)
- numpy (audio processing)

✅ **VB-Audio Virtual Cable**
- Virtual audio device driver
- Enables Discord/Game audio routing
- Downloaded and installed automatically

## 📋 Manual Installation (If auto-install fails)

### Step 1: Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install VB-Cable manually
1. Download: https://vb-audio.com/Cable/
2. Extract and run VBCABLE_Setup_x64.exe
3. Restart computer

## 🎮 First Run

After installation:

```bash
python main.py
```

Then:
1. Click "➕ Add Sound" to add audio files
2. Click "⚙️ Audio Setup" to configure routing
3. Select "CABLE Input" device
4. Click "▶️ Start Routing"
5. Configure Discord/Game to use "CABLE Output" as microphone

## 🔧 Troubleshooting

### PyAudio installation fails

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install python3-pyaudio
```

### VB-Cable installation fails

1. Download manually: https://vb-audio.com/Cable/
2. Right-click setup.exe → Run as Administrator
3. Restart computer

### "Need administrator privileges" error

Right-click `setup.bat` → Run as Administrator

## 🎯 Verification

After installation, verify everything works:

```bash
python -c "import pygame; import pyaudio; import numpy; print('✅ All dependencies installed!')"
```

Check VB-Cable:
1. Run `python main.py`
2. Click "⚙️ Audio Setup"
3. You should see "CABLE Input" in the device list

## 📦 What's Installed Where?

- **Python packages**: In your Python environment
- **VB-Cable driver**: `C:\Program Files\VB\CABLE\`
- **Soundboard files**: Current directory
- **Audio files**: `./sounds/` folder

## 🗑️ Uninstallation

### Remove Python packages
```bash
pip uninstall pygame-ce pyaudio numpy
```

### Remove VB-Cable
1. Go to: `C:\Program Files\VB\CABLE\`
2. Run: `VBCABLE_Setup_x64.exe`
3. Click "Remove Driver"
4. Restart computer

## 💡 Tips

- **First time setup**: Takes ~5 minutes
- **Requires restart**: After VB-Cable installation
- **Admin rights**: Needed for driver installation
- **Internet required**: For downloading VB-Cable

## 🆘 Still Having Issues?

1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions
2. Check [QUICK_START.md](QUICK_START.md) for quick reference
3. Make sure Python 3.7+ is installed
4. Try manual installation steps above

## 🎉 Success!

Once installed, you're ready to:
- Play sounds with one click
- Route audio to Discord/Games
- Troll your friends with meme sounds
- Add professional sound effects to streams

Enjoy your Soundboard Pro! 🎵
