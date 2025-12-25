import numpy as np
import logging

# Try imports
try:
    import pygame
except ImportError:
    try:
        import pygame_ce as pygame
    except ImportError:
        pygame = None

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AudioUtils:
    """Utilities for audio loading and processing."""
    
    INT16_NORMALIZATION = 32768.0
    INT32_NORMALIZATION = 2147483648.0
    
    @staticmethod
    def load_and_convert_audio(path: str):
        """Load audio file and convert to float32 array
        
        Returns:
            tuple: (audio_data, frequency) or (None, 0)
        """
        if not pygame or not pygame.mixer.get_init():
            return None, 0
            
        try:
            snd = pygame.mixer.Sound(path)
            arr = pygame.sndarray.array(snd)
            freq = pygame.mixer.get_init()[0]
            
            # Convert to float32
            if arr.dtype == np.int16:
                audio = arr.astype(np.float32) / AudioUtils.INT16_NORMALIZATION
            elif arr.dtype == np.int32:
                audio = arr.astype(np.float32) / AudioUtils.INT32_NORMALIZATION
            else:
                audio = arr.astype(np.float32)
                
            return audio, freq
        except Exception as e:
            logger.error(f"Error loading audio file {path}: {e}")
            return None, 0

    @staticmethod
    def process_audio(audio, freq, trim_start=0.0, trim_end=0.0, pitch=1.0, target_samplerate=None):
        """Apply trim, pitch, resampling and tiny fades to audio data"""
        if audio is None or len(audio) == 0:
            return audio
            
        # 1. Apply trim
        if trim_start > 0 or trim_end > 0:
            start_sample = int(trim_start * freq)
            end_sample = int(trim_end * freq) if trim_end > 0 else len(audio)
            
            start_sample = max(0, min(start_sample, len(audio)))
            end_sample = max(start_sample, min(end_sample, len(audio)))
            
            audio = audio[start_sample:end_sample]
            
        if len(audio) == 0:
            return audio

        # 2. Resample if needed (High Quality)
        if target_samplerate and (pitch != 1.0 or freq != target_samplerate):
             pitch_factor = 1.0 / pitch if pitch != 1.0 else 1.0
             rate_factor = target_samplerate / freq if freq != target_samplerate else 1.0
             final_length = int(len(audio) * pitch_factor * rate_factor)

             if final_length > 0:
                if SCIPY_AVAILABLE:
                    # High quality resampling
                    audio = signal.resample(audio, final_length).astype(np.float32)
                else:
                    # Fallback to linear interpolation
                    if audio.ndim == 1:
                        indices = np.linspace(0, len(audio) - 1, final_length)
                        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
                    else:
                        indices = np.linspace(0, len(audio) - 1, final_length)
                        resampled = []
                        for ch in range(audio.shape[1]):
                            resampled.append(np.interp(indices, np.arange(len(audio)), audio[:, ch]))
                        audio = np.column_stack(resampled).astype(np.float32)
                
        # 3. Add tiny fades (5ms) to prevent clicks at start/end
        if len(audio) > 100:
            fade_len = min(int(freq * 0.005), len(audio) // 10)
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
                fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
                
                if audio.ndim == 1:
                    audio[:fade_len] *= fade_in
                    audio[-fade_len:] *= fade_out
                else:
                    # Apply to each channel
                    fade_in = fade_in.reshape(-1, 1)
                    fade_out = fade_out.reshape(-1, 1)
                    audio[:fade_len] *= fade_in
                    audio[-fade_len:] *= fade_out

        return audio

    @staticmethod
    def get_audio_duration(path: str) -> float:
        """Get audio file duration in seconds"""
        if not pygame or not pygame.mixer.get_init():
            return 0.0
            
        try:
            snd = pygame.mixer.Sound(path)
            return snd.get_length()
        except Exception as e:
            logger.error(f"Error getting duration for {path}: {e}")
            return 0.0

    @staticmethod
    def get_waveform_data(path: str, samples: int = 200) -> list:
        """Generate waveform data for visualization"""
        audio, _ = AudioUtils.load_and_convert_audio(path)
        
        if audio is None:
            return []
            
        try:
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Downsample to requested number of samples
            chunk_size = max(1, len(audio) // samples)
            waveform = []
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) > 0:
                    # Get peak amplitude in this chunk
                    peak = float(np.max(np.abs(chunk)))
                    waveform.append(peak)
                    if len(waveform) >= samples:
                        break
            
            return waveform
        except Exception as e:
            logger.error(f"Error generating waveform for {path}: {e}")
            return []
