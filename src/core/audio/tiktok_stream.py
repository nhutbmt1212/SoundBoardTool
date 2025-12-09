"""TikTok Streaming - Stream TikTok audio to VB-Cable"""
import sys
import os

# Import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.config_paths import get_tiktok_cache_dir
from core.audio.stream_base import BaseStream

class TikTokStream(BaseStream):
    """Handles TikTok audio streaming to VB-Cable with persistent cache"""
    
    def __init__(self, vb_manager):
        super().__init__(vb_manager, get_tiktok_cache_dir())
        self.ytdl_format = 'bestaudio/best'
