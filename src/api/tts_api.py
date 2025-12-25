"""API Layer - Text-to-Speech Endpoints using Edge TTS"""
import eel
import asyncio
import tempfile
import os
import threading
import hashlib
import edge_tts


class TTSAPI:
    """Text-to-Speech API endpoints using Microsoft Edge TTS"""
    
    # Vietnamese voices available
    VOICES = {
        'vi-VN-HoaiMyNeural': 'Hoài My (Nữ)',
        'vi-VN-NamMinhNeural': 'Nam Minh (Nam)'
    }
    
    # Cache directory for TTS files
    CACHE_DIR = os.path.join(tempfile.gettempdir(), 'dalit_tts_cache')
    
    def __init__(self, audio_engine):
        self.audio = audio_engine
        self.cached_file = None
        self.cached_hash = None
        self._lock = threading.Lock()
        self._cancel_flag = threading.Event()
        self._is_generating = False
        self._ensure_cache_dir()
        self._register_endpoints()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
    
    def _register_endpoints(self):
        """Register all TTS endpoints with Eel"""
        eel.expose(self.generate_and_play_tts)
        eel.expose(self.get_tts_voices)
        eel.expose(self.stop_tts)
        eel.expose(self.cancel_tts)
        eel.expose(self.is_tts_generating)
    
    def _get_cache_hash(self, text: str, voice: str) -> str:
        """Generate hash from text and voice for caching"""
        content = f"{text}:{voice}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _cleanup_old_cache(self, new_hash: str):
        """Delete old cache file if hash changed"""
        with self._lock:
            if self.cached_file and self.cached_hash != new_hash:
                if os.path.exists(self.cached_file):
                    try:
                        os.remove(self.cached_file)
                        print(f"[TTS] Removed old cache: {self.cached_file}")
                    except Exception as e:
                        print(f"[TTS] Failed to remove old cache: {e}")
                self.cached_file = None
                self.cached_hash = None
    
    def generate_and_play_tts(self, text: str, voice: str = 'vi-VN-HoaiMyNeural', volume: float = 1.0):
        """Generate speech from text and play through VB-Cable"""
        try:
            if not text or not text.strip():
                return {'success': False, 'error': 'Vui lòng nhập văn bản'}
            
            # Reset cancel flag
            self._cancel_flag.clear()
            self._is_generating = True
            
            text = text.strip()
            current_hash = self._get_cache_hash(text, voice)
            cache_file = os.path.join(self.CACHE_DIR, f"tts_{current_hash}.mp3")
            
            # Check if we can use cached file
            if self.cached_hash == current_hash and self.cached_file and os.path.exists(self.cached_file):
                print(f"[TTS] Using cached file: {self.cached_file}")
            else:
                # Clean up old cache if different
                self._cleanup_old_cache(current_hash)
                
                print(f"[TTS] Generating speech for: {text[:50]}...")
                
                # Run edge-tts in a separate thread
                generation_error = [None]
                
                def run_tts():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            communicate = edge_tts.Communicate(text, voice)
                            loop.run_until_complete(communicate.save(cache_file))
                        finally:
                            loop.close()
                    except Exception as e:
                        generation_error[0] = str(e)
                
                thread = threading.Thread(target=run_tts)
                thread.start()
                
                # Wait with cancel check
                while thread.is_alive():
                    if self._cancel_flag.is_set():
                        print("[TTS] Generation cancelled")
                        self._is_generating = False
                        return {'success': False, 'error': 'Đã hủy', 'cancelled': True}
                    thread.join(timeout=0.1)
                
                if self._cancel_flag.is_set():
                    self._is_generating = False
                    # Clean up partial file
                    if os.path.exists(cache_file):
                        try:
                            os.remove(cache_file)
                        except:
                            pass
                    return {'success': False, 'error': 'Đã hủy', 'cancelled': True}
                
                if generation_error[0]:
                    self._is_generating = False
                    raise Exception(generation_error[0])
                
                if not os.path.exists(cache_file) or os.path.getsize(cache_file) == 0:
                    self._is_generating = False
                    raise Exception("Không thể tạo file âm thanh. Kiểm tra kết nối mạng.")
                
                # Update cache info
                with self._lock:
                    self.cached_file = cache_file
                    self.cached_hash = current_hash
                
                print(f"[TTS] Audio generated: {cache_file}")
            
            self._is_generating = False
            
            # Verify file exists before playing
            if not self.cached_file or not os.path.exists(self.cached_file):
                return {'success': False, 'error': 'File không tồn tại'}
            
            # Stop any current playback
            self.audio.stop()
            
            # Set volume with headroom (max 90% of requested to avoid clipping after processing)
            safe_volume = min(volume * 0.9, 1.0)
            self.audio.set_volume(safe_volume)
            
            # Play the cached file
            self.audio.sound_player.play_file(self.cached_file)
            
            return {'success': True}
            
        except Exception as e:
            self._is_generating = False
            print(f"[TTS] Error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_tts_voices(self):
        """Get available TTS voices"""
        return self.VOICES
    
    def cancel_tts(self):
        """Cancel TTS generation"""
        self._cancel_flag.set()
        self._is_generating = False
        print("[TTS] Cancel requested")
        return {'success': True}
    
    def is_tts_generating(self):
        """Check if TTS is currently generating"""
        return self._is_generating
    
    def stop_tts(self):
        """Stop TTS playback"""
        self._cancel_flag.set()
        self._is_generating = False
        self.audio.stop()
