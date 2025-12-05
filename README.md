# 🎵 Soundboard Pro

Professional soundboard tool with TRUE standalone executable - no Python or dependencies needed!

## ✨ Features

- 🎨 **Beautiful Modern UI** - Dark theme with custom buttons
- 🎵 **Easy Sound Management** - Add and play sounds with one click
- 🔊 **Volume Control** - Adjust volume in real-time
- 🎙️ **Audio Routing** - Route sounds to Discord/Games
- 🚀 **TRUE Standalone** - No Python installation needed
- 💾 **Single EXE File** - Everything bundled

## 🚀 For Users

### Download & Run
1. Download `SoundboardPro.exe` from [Releases](https://github.com/yourusername/soundboard-pro/releases)
2. Double-click to run
3. First run: Optionally install VB-Cable for Discord/Game routing
4. That's it! No setup needed!

### Auto Install (From Source)
```bash
setup.bat
```
**Tự động 100%:**
- ✅ Tự động yêu cầu quyền Admin
- ✅ Tự động cài Python (nếu chưa có)
- ✅ Tự động cài thư viện
- ✅ Tự động tải và cài VB-Cable
- ✅ Thử 4 phương pháp cài đặt khác nhau

**Xem chi tiết:** [AUTO_INSTALL.md](AUTO_INSTALL.md)

### Features
- Add sounds (.wav, .mp3, .ogg, .flac)
- Play sounds with one click
- Adjust volume
- Route audio to Discord/Games (with VB-Cable)
- Stop all sounds

## 🔧 For Developers

### Build Standalone EXE

```bash
# Quick build
build.bat

# Or manual
pip install pyinstaller pygame-ce pyaudio numpy
python build_exe.py
```

**Output:** `dist/SoundboardPro.exe` (~40-50MB)

### What Gets Bundled
- ✅ Python runtime
- ✅ pygame-ce (audio playback)
- ✅ pyaudio (audio routing)
- ✅ numpy (audio processing)
- ✅ tkinter (GUI)
- ✅ VB-Cable installer
- ✅ All source code

### Development Setup

```bash
# Clone repo
git clone https://github.com/yourusername/soundboard-pro.git
cd soundboard-pro

# Install dependencies
pip install -r requirements.txt

# Run from source
python src/main.py
```

## 📁 Project Structure

```
soundboard-pro/
├── src/                    # Source code
│   ├── main.py            # Entry point
│   ├── main_standalone.py # Standalone entry point
│   ├── ui.py              # GUI
│   ├── soundboard.py      # Audio logic
│   ├── audio_router.py    # Routing system
│   └── config.py          # Configuration
│
├── sounds/                 # Audio files
├── scripts/                # Setup scripts (for development)
├── build_exe.py           # Build script
├── build.bat              # Build wrapper
└── BUILD_STANDALONE.md    # Build documentation
```

## 🎮 Use Cases

- Gaming with friends
- Discord trolling with meme sounds
- Streaming with sound effects
- Presentations
- Podcasting

## 📖 Documentation

- [Build Guide](BUILD_STANDALONE.md) - How to build standalone EXE
- [Changelog](CHANGELOG.md) - Version history
- [Contributing](CONTRIBUTING.md) - How to contribute

## 🔧 Requirements

### For Users:
- Windows 10/11
- Nothing else! (Everything bundled in EXE)

### For Developers:
- Python 3.7+
- PyInstaller
- pygame-ce, pyaudio, numpy

## 🆘 Troubleshooting

### EXE doesn't start
- Run as Administrator
- Check antivirus (may flag as false positive)

### VB-Cable not installing
- Run EXE as Administrator
- Or download manually: https://vb-audio.com/Cable/

### Discord doesn't hear sounds
- Check Input Device = "CABLE Output"
- Ensure VB-Cable is installed
- Restart Discord

## 📝 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## 🌟 Credits

- Python & PyInstaller
- pygame community
- VB-Audio Software

---

**Made with ❤️ | Happy Sound Boarding! 🎵**

**Download:** [Latest Release](https://github.com/yourusername/soundboard-pro/releases)
