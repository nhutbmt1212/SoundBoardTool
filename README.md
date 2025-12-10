# 🎵 Soundboard Pro

Soundboard application với VB-Cable support cho Discord/streaming.

## Download & Run

1. Download `SoundboardPro.exe` từ [Releases](../../releases)
2. Chạy file exe
3. Done! App tự động mở trong browser

## Features

- 🎹 Keybind cho từng sound (Shift+1, Ctrl+F1, etc.)
- 🔊 Volume riêng cho từng sound
- 🎙️ VB-Cable routing cho Discord/OBS
- 🌐 Modern web UI
- 🎬 YouTube/TikTok streaming support
- 🎚️ Audio effects (reverb, echo, etc.)
- 🔄 Loop functionality
- ✂️ Trim support

## VB-Cable (Optional)

Để stream sound qua Discord:
1. Tải [VB-Cable](https://vb-audio.com/Cable/)
2. Cài đặt và restart
3. Discord → Settings → Voice → Input Device → "CABLE Output"

## Build từ source

```bash
pip install -r requirements.txt
python build_exe.py
```

Output: `dist/SoundboardPro.exe`

## Dev

```bash
pip install -r requirements.txt
python src/app.py
```

## 📚 Architecture Documentation

For developers working on this project, see:
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Quick architecture overview
- **Workflows**: Use `/architecture` for quick reference
- **Full docs**: See `.gemini/antigravity/brain/` for detailed documentation

## Structure

```
├── src/
│   ├── app.py          # Entry point
│   ├── core/
│   │   ├── audio/      # Audio engine
│   │   └── config.py   # Config
│   ├── api/            # API endpoints
│   └── web/            # Frontend
├── sounds/             # Sound files
└── build_exe.py        # Build script
```

