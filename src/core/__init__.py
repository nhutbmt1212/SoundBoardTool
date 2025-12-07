# Core modules
from .audio.audio_engine import AudioEngine
from .config import Config
from .hotkey import hotkey_manager

__all__ = ['AudioEngine', 'Config', 'hotkey_manager']

# AudioEngine is a facade class combining:
# - VBCableManager: VB-Cable device detection
# - SoundPlayer: Sound file playback
# - MicPassthrough: Mic routing to VB-Cable
# - YouTubeStream: YouTube audio streaming
