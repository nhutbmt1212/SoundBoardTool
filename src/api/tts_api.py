"""API Layer - Text-to-Speech Endpoints using Edge TTS"""
import eel
import asyncio
import tempfile
import os
import threading
import edge_tts


class TTSAPI:
    """Text-to-Speech API endpoints using Microsoft Edge TTS"""
    
    # Vietnamese voices available
    VOICES = {
        'vi-VN-HoaiMyNeural': 'Hoài My (Nữ)',
        'vi-VN-NamMinhNeural': 'Nam Minh (Nam)'
    }
    
    def __init__(self, audio_engine):
        self.audio = audio_engine
        self.temp_file = None
        self._lock = threading.Lock()
        self._register_endpoints()
    
    def _register_endpoints(self):
        """Register all TTS endpoints with Eel"""
        eel.expose(self.generate_and_play_tts)
        eel.expose(self.get_tts_voices)
        eel.expose(self.stop_tts)
    
    def _cleanup_temp_file(self):
        """Clean up the temporary TTS audio file"""
        with self._lock:
            if self.temp_file and os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                    print(f"[TTS] Cleaned up temp file: {self.temp_file}")
                except Exception as e:
                    print(f"[TTS] Failed to cleanup temp file: {e}")
                finally:
                    self.temp_file = None
    
    async def _generate_audio_async(self, text: str, voice: str, output_path: str):
        """Async method to generate TTS audio"""
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
    
    def generate_and_play_tts(self, text: str, voice: str = 'vi-VN-HoaiMyNeural', volume: float = 1.0):
        """Generate speech from text and play through VB-Cable
        
        Args:
            text: Text to convert to speech
            voice: Edge TTS voice name
            volume: Playback volume (0.0 to 1.0)
            
        Returns:
            dict: {'success': bool, 'error': str (optional)}
        """
        try:
            if not text or not text.strip():
                return {'success': False, 'error': 'Vui lòng nhập văn bản'}
            
            # Cleanup previous temp file if exists
            self._cleanup_temp_file()
            
            # Create new temp file
            with self._lock:
                fd, self.temp_file = tempfile.mkstemp(suffix='.mp3', prefix='tts_')
                os.close(fd)
            
            print(f"[TTS] Generating speech for text: {text[:50]}...")
            
            # Run edge-tts in a separate thread with its own event loop
            generation_error = [None]
            
            def run_tts():
                try:
                    import asyncio
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        communicate = edge_tts.Communicate(text, voice)
                        loop.run_until_complete(communicate.save(self.temp_file))
                    finally:
                        loop.close()
                except Exception as e:
                    generation_error[0] = str(e)
            
            # Run in thread and wait
            thread = threading.Thread(target=run_tts)
            thread.start()
            thread.join(timeout=30)  # 30 second timeout
            
            if generation_error[0]:
                raise Exception(generation_error[0])
            
            # Check if file was created and has content
            if not os.path.exists(self.temp_file) or os.path.getsize(self.temp_file) == 0:
                raise Exception("Không thể tạo file âm thanh. Kiểm tra kết nối mạng.")
            
            print(f"[TTS] Audio generated: {self.temp_file}")
            
            # Stop any current playback
            self.audio.stop()
            
            # Set volume and play using sound player
            self.audio.set_volume(volume)
            
            # Play the temp file directly through sound player
            self.audio.sound_player.play_file(self.temp_file, on_complete=self._cleanup_temp_file)
            
            return {'success': True}
            
        except Exception as e:
            print(f"[TTS] Error: {e}")
            self._cleanup_temp_file()
            return {'success': False, 'error': str(e)}
    
    def get_tts_voices(self):
        """Get available TTS voices
        
        Returns:
            dict: Voice ID to display name mapping
        """
        return self.VOICES
    
    def stop_tts(self):
        """Stop TTS playback and cleanup"""
        self.audio.stop()
        self._cleanup_temp_file()
