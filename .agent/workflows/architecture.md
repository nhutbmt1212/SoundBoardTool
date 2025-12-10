---
description: Quick reference for SoundBoardTool architecture
---

# SoundBoardTool Architecture Quick Reference

This workflow provides quick access to architecture documentation for faster code navigation.

## Documentation Files

1. **Full Architecture Documentation**
   - Location: `C:\Users\nhutb\.gemini\antigravity\brain\7bf85401-ece0-423b-ae27-445d403b17ac\soundboard_architecture.md`
   - Contains: Complete architecture overview, flow diagrams, component breakdown, data flow sequences

2. **Quick Reference Guide**
   - Location: `C:\Users\nhutb\.gemini\antigravity\brain\7bf85401-ece0-423b-ae27-445d403b17ac\quick_reference.md`
   - Contains: Cheat sheet for fast navigation, file paths, key functions

## Quick Navigation by Feature

### Sound Playback
- **Backend**: `src/core/audio/audio_engine.py` → `src/core/audio/sound_player.py`
- **API**: `src/api/sound_api.py`
- **Frontend**: `src/web/js/events/sound.js` → `src/web/js/ui/card-renderer.js`

### YouTube Streaming
- **Backend**: `src/core/audio/stream_base.py` → `src/core/audio/youtube_stream.py`
- **API**: `src/api/youtube_api.py`
- **Frontend**: `src/web/js/events/youtube.js` → `src/web/js/ui/panel-renderer.js`

### TikTok Streaming
- **Backend**: `src/core/audio/stream_base.py` → `src/core/audio/tiktok_stream.py`
- **API**: `src/api/tiktok_api.py`
- **Frontend**: `src/web/js/events/tiktok.js` → `src/web/js/ui/panel-renderer.js`

### Audio Effects
- **Backend**: `src/core/audio/effects_processor.py`
- **Frontend**: `src/web/js/events/effects-events.js` → `src/web/js/ui/effects-helpers.js`

### Hotkeys
- **Backend**: `src/core/hotkey.py` → `src/services/hotkey_service.py`
- **Frontend**: `src/web/js/events/keybind.js`

### Settings/Config
- **Backend**: `src/core/config.py` → `src/api/settings_api.py`
- **Frontend**: `src/web/js/core/state.js` → `src/web/js/core/api.js`

## Key Entry Points

- **Backend Entry**: `src/app.py` - `main()` function
- **Frontend Entry**: `src/web/js/core/app.js` - `init()` function
- **Audio Facade**: `src/core/audio/audio_engine.py` - `AudioEngine` class
- **State Management**: `src/web/js/core/state.js` - `AppState` object

## Common Tasks

### Add New Sound Effect
1. Backend: `src/core/audio/effects_processor.py` - Add effect method
2. Frontend: `src/web/js/events/effects-events.js` - Add UI control
3. UI: `src/web/js/ui/effects-helpers.js` - Add effect UI

### Add New API Endpoint
1. Backend: API file (e.g., `src/api/sound_api.py`) - Add method + `eel.expose()`
2. Frontend: `src/web/js/core/api.js` - Add wrapper function

### Modify UI Component
1. Renderer: `src/web/js/ui/card-renderer.js` or `panel-renderer.js`
2. Events: `src/web/js/events/events.js` - Add event handler
3. State: `src/web/js/core/state.js` - Update state if needed

### Debug Audio Issue
1. Check: `src/core/audio/audio_engine.py` - `play()` method
2. Check: `src/core/audio/vb_cable_manager.py` - VB-Cable connection
3. Check: `src/core/audio/stream_base.py` - `_stream_loop()` for streaming

## Architecture Overview

```
SoundBoardTool Architecture:

Entry Point (app.py)
    ↓
AudioEngine (Facade)
    ├── SoundPlayer (local files)
    ├── YouTubeStream (YouTube)
    ├── TikTokStream (TikTok)
    ├── MicPassthrough (mic routing)
    ├── EffectsProcessor (effects)
    └── VBCableManager (VB-Cable)
    ↓
API Layer (Eel Bridge)
    ├── SoundAPI
    ├── YouTubeAPI
    ├── TikTokAPI
    └── SettingsAPI
    ↓
Frontend (JavaScript)
    ├── app.js (entry)
    ├── AppState (state)
    ├── EventHandlers (events)
    └── UI Renderers (rendering)
```

## Usage

When working on SoundBoardTool:
1. Identify the feature you're working on
2. Use this guide to find relevant files
3. Refer to full documentation for detailed flow diagrams
4. Use quick reference for common patterns

## Notes

- All documentation is auto-generated from code analysis
- Update this workflow when adding major new features
- Keep architecture docs in sync with code changes
