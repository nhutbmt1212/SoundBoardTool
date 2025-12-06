# 🎵 Soundboard Pro

A modern soundboard application with VB-Cable support for Discord/streaming.

## Features

- 🎹 Customizable keybinds per sound
- 🔊 Individual volume control
- 🎙️ VB-Cable routing for Discord/OBS
- 🌐 Modern web-based UI

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python src/app.py
```

## Build Executable

```bash
python build_exe.py
```

Output: `dist/SoundboardPro.exe`

## VB-Cable Setup

1. Download from [vb-audio.com/Cable](https://vb-audio.com/Cable/)
2. Install and restart
3. In Discord: Settings → Voice → Input Device → "CABLE Output"

## Project Structure

```
├── src/
│   ├── app.py          # Main application
│   ├── core/
│   │   ├── audio.py    # Audio engine
│   │   └── config.py   # Configuration
│   └── web/            # Frontend (HTML/CSS/JS)
├── sounds/             # Sound files
├── requirements.txt
└── build_exe.py
```

## Requirements

- Python 3.10+
- Chrome or Edge browser
- VB-Cable (optional, for Discord)
