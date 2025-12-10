---
description: Navigate to specific component quickly
---

# Quick Component Navigation

Use this workflow to quickly jump to specific components in the codebase.

## Backend Components

### Core Audio
```bash
# Audio Engine (Main Facade)
code src/core/audio/audio_engine.py

# Sound Player (Local files)
code src/core/audio/sound_player.py

# Stream Base (YouTube/TikTok base)
code src/core/audio/stream_base.py

# YouTube Streaming
code src/core/audio/youtube_stream.py

# TikTok Streaming
code src/core/audio/tiktok_stream.py

# Effects Processor
code src/core/audio/effects_processor.py

# Mic Passthrough
code src/core/audio/mic_passthrough.py

# VB-Cable Manager
code src/core/audio/vb_cable_manager.py
```

### API Layer
```bash
# Sound API
code src/api/sound_api.py

# YouTube API
code src/api/youtube_api.py

# TikTok API
code src/api/tiktok_api.py

# Settings API
code src/api/settings_api.py
```

### Services
```bash
# Hotkey Service
code src/services/hotkey_service.py

# Hotkey Core
code src/core/hotkey.py
```

### Configuration
```bash
# Config Management
code src/core/config.py

# Config Paths
code src/core/config_paths.py
```

## Frontend Components

### Core
```bash
# App Entry Point
code src/web/js/core/app.js

# State Management
code src/web/js/core/state.js

# API Client
code src/web/js/core/api.js

# Utilities
code src/web/js/core/utils.js
```

### Event Handlers
```bash
# Main Event Delegator
code src/web/js/events/events.js

# Sound Events
code src/web/js/events/sound.js

# YouTube Events
code src/web/js/events/youtube.js

# TikTok Events
code src/web/js/events/tiktok.js

# Keybind Events
code src/web/js/events/keybind.js

# Effects Events
code src/web/js/events/effects-events.js

# UI Events
code src/web/js/events/ui.js
```

### UI Renderers
```bash
# Main UI Coordinator
code src/web/js/ui/ui.js

# Card Renderer
code src/web/js/ui/card-renderer.js

# Grid Renderer
code src/web/js/ui/grid-renderer.js

# Panel Renderer
code src/web/js/ui/panel-renderer.js

# UI Helpers
code src/web/js/ui/helpers.js

# Effects Helpers
code src/web/js/ui/effects-helpers.js
```

### Features
```bash
# Waveform (Sounds)
code src/web/js/features/waveform.js

# YouTube Waveform
code src/web/js/features/youtube-waveform.js

# TikTok Waveform
code src/web/js/features/tiktok-waveform.js
```

## Entry Points

```bash
# Backend Entry
code src/app.py

# Frontend Entry
code src/web/js/core/app.js

# Main HTML
code src/web/index.html

# Main CSS
code src/web/style.css
```

## Usage

To open a specific component, copy the command and run it in your terminal, or use your IDE's file navigation with the path.
