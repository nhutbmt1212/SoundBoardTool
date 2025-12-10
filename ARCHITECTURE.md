# SoundBoardTool - Architecture & Code Navigation

## 📚 Documentation

This project has comprehensive architecture documentation to help you navigate the codebase quickly.

### Quick Access

- **Architecture Overview**: Use `/architecture` workflow for quick reference
- **Full Documentation**: See `.gemini/antigravity/brain/` for detailed docs

### Key Documents

1. **soundboard_architecture.md** - Complete architecture guide with:
   - System overview and technology stack
   - Architecture flow diagrams (Mermaid)
   - Component breakdown (30+ Python files, 23+ JS files)
   - Data flow sequences
   - Navigation guide by feature
   - Common patterns and debugging tips

2. **quick_reference.md** - Fast lookup cheat sheet with:
   - File paths by feature
   - Key functions and classes
   - Common task shortcuts
   - Search patterns

### Project Structure

```
SoundBoardTool/
├── src/
│   ├── app.py                    # 🚀 Backend Entry Point
│   ├── core/audio/               # Audio processing core
│   │   ├── audio_engine.py       # Main audio facade
│   │   ├── stream_base.py        # Streaming base class
│   │   ├── youtube_stream.py     # YouTube streaming
│   │   ├── tiktok_stream.py      # TikTok streaming
│   │   ├── sound_player.py       # Local sound playback
│   │   ├── effects_processor.py  # Audio effects
│   │   └── ...
│   ├── api/                      # API endpoints (Eel)
│   ├── services/                 # Services (hotkeys, etc.)
│   └── web/                      # Frontend
│       └── js/
│           ├── core/             # Core modules
│           │   └── app.js        # 🚀 Frontend Entry Point
│           ├── events/           # Event handlers
│           ├── ui/               # UI renderers
│           └── features/         # Features (waveforms, etc.)
└── .agent/workflows/             # Antigravity workflows
    └── architecture.md           # Quick architecture reference
```

### Quick Navigation by Feature

| Feature | Backend | API | Frontend |
|---------|---------|-----|----------|
| **Sound Playback** | `core/audio/sound_player.py` | `api/sound_api.py` | `events/sound.js` |
| **YouTube** | `core/audio/youtube_stream.py` | `api/youtube_api.py` | `events/youtube.js` |
| **TikTok** | `core/audio/tiktok_stream.py` | `api/tiktok_api.py` | `events/tiktok.js` |
| **Effects** | `core/audio/effects_processor.py` | - | `events/effects-events.js` |
| **Hotkeys** | `services/hotkey_service.py` | - | `events/keybind.js` |
| **Settings** | `core/config.py` | `api/settings_api.py` | `core/state.js` |

### Architecture Pattern

```
User Interaction
    ↓
Frontend (JavaScript)
    ↓
Eel Bridge (Python ↔ JS)
    ↓
API Layer
    ↓
AudioEngine (Facade)
    ↓
Core Components (Sound, Stream, Effects, etc.)
    ↓
Audio Output (Speaker + VB-Cable)
```

### Common Tasks

- **Add new feature**: Start with `audio_engine.py` (backend) and `events.js` (frontend)
- **Modify UI**: Check `ui/` folder for renderers
- **Debug audio**: Check `audio_engine.py` → specific component
- **Add API endpoint**: Add to API file + expose with Eel + add to `api.js`

### Development

```bash
# Run development server
python src/app.py

# Build executable
python build_exe.py
```

### Code Quality

- ✅ Clean code principles enforced
- ✅ Modular architecture (SRP, DRY)
- ✅ Type hints and docstrings
- ✅ No magic numbers (constants extracted)

---

For detailed information, use the `/architecture` workflow or refer to the full documentation in the `.gemini` directory.
