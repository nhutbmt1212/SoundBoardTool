# 🎵 Soundboard Pro

Professional soundboard tool with automatic installation and Discord/Game audio routing.

## 🚀 Quick Start

```bash
# 1. Setup (first time only)
setup.bat

# 2. Restart computer

# 3. Run app
run.bat
```

That's it! Everything is automated.

## ✨ Features

- ✅ **Auto-install Python** - Detects and installs Python 3.11.7 if needed
- ✅ **Auto-install VB-Cable** - Downloads and installs virtual audio driver
- ✅ **Beautiful UI** - Modern dark theme with custom buttons
- ✅ **Audio Routing** - Route sounds to Discord/Games
- ✅ **Easy to Use** - Add sounds and play with one click
- ✅ **Volume Control** - Adjust volume in real-time

## 📁 Project Structure

```
soundboard/
├── setup.bat           # Run once to install everything
├── run.bat             # Run daily to start app
├── requirements.txt    # Python dependencies
│
├── src/                # Source code
│   ├── main.py         # Entry point
│   ├── ui.py           # GUI interface
│   ├── soundboard.py   # Audio logic
│   ├── audio_router.py # Routing system
│   └── config.py       # Configuration
│
├── scripts/            # Installation scripts
│   ├── setup.bat       # Main setup script
│   ├── run.bat         # Run script
│   ├── setup.py        # Python setup
│   ├── installer.py    # VB-Cable installer
│   ├── python_installer.py
│   ├── install_python.bat
│   └── check_python.bat
│
├── docs/               # Documentation
│   ├── README.md       # Full documentation
│   ├── QUICK_START.md  # Quick reference
│   ├── INSTALL.md      # Installation guide
│   └── SETUP_GUIDE.md  # Audio setup guide
│
└── sounds/             # Your audio files
```

## 📖 Documentation

- [Quick Start](docs/QUICK_START.md) - Get started in 3 minutes
- [Installation Guide](docs/INSTALL.md) - Detailed installation instructions
- [Setup Guide](docs/SETUP_GUIDE.md) - Audio routing configuration
- [Full Documentation](docs/README.md) - Complete reference

## 🎮 Use Cases

- Gaming with friends
- Discord trolling with meme sounds
- Streaming with sound effects
- Presentations
- Podcasting

## 🔧 Requirements

- Windows 10/11
- Python 3.7+ (auto-installed)
- Internet connection (for setup)
- Admin rights (for driver installation)

## 🆘 Troubleshooting

### Python not found
```bash
scripts\install_python.bat
```

### Setup fails
```bash
# Run as Administrator
Right-click setup.bat → Run as Administrator
```

### Can't find CABLE device
- Restart computer after setup
- Check docs/SETUP_GUIDE.md

### Discord doesn't hear sounds
- Check Input Device = "CABLE Output"
- See docs/SETUP_GUIDE.md for details

## 📝 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

Made with ❤️ | Happy Sound Boarding! 🎵
