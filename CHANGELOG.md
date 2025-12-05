# Changelog

All notable changes to Soundboard Pro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-12-05

### Fixed
- Python installation no longer hangs on GUI installer
- Removed Y/N confirmation prompts - fully automatic now
- Fixed setup hanging at "Please check 'Add Python to PATH'"
- Improved progress indicators with clear step numbers
- Better error messages with specific solutions

### Changed
- Python installs silently with `/quiet` flag
- Auto-exits after Python install (5 second timeout)
- Clear status icons: [OK], [!], [X], [*]
- Step indicators: [1/3], [2/3], [3/3]
- Removed all user confirmation prompts

### Improved
- Better error handling in installer
- Clearer next steps after installation
- More descriptive progress messages
- Automatic continuation where possible

## [1.0.0] - 2025-12-05

### Added
- Initial release
- Auto Python installation system
- Auto VB-Cable installation system
- Modern dark theme UI
- Audio routing to Discord/Games
- Volume control
- Sound management (add, play, stop)
- Comprehensive documentation
- One-click setup and run
- Virtual device auto-detection
- Beautiful custom buttons with hover effects
- Scrollable sound grid layout

### Features
- Zero manual configuration required
- Automatic dependency installation
- Professional UI design
- Real-time volume adjustment
- Audio routing status indicator
- Support for .wav, .mp3, .ogg, .flac formats

### Documentation
- Quick Start guide
- Installation guide
- Setup guide for audio routing
- Full README with examples
- Contributing guidelines
- MIT License

## [Unreleased]

### Planned for 1.1.0
- Hotkeys support
- Sound categories/folders
- Favorites system
- Search/filter sounds
- Sound preview

### Planned for 1.2.0
- Custom button colors
- Waveform visualization
- Export/import sound packs
- Themes support

### Planned for 2.0.0
- Multi-language support
- Cloud sync
- Mobile companion app
- Plugin system
- Advanced audio effects

---

[1.0.0]: https://github.com/yourusername/soundboard-pro/releases/tag/v1.0.0
