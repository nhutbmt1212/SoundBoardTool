"""Audio Effects Processor - Real-time audio effects using scipy"""
import numpy as np
import logging
from scipy import signal
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AudioEffectsProcessor:
    """Real-time audio effects processor using scipy signal processing"""
    
    # Constants
    DEFAULT_SAMPLERATE = 48000
    MAX_DELAY_SECONDS = 2.0
    REVERB_IR_LENGTH = 24000  # 0.5s at 48kHz
    
    def __init__(self, samplerate: int = DEFAULT_SAMPLERATE):
        """Initialize effects processor
        
        Args:
            samplerate: Audio sample rate in Hz
        """
        self.samplerate = samplerate
        
        # Delay buffer for echo effect (circular buffer)
        max_delay_samples = int(self.MAX_DELAY_SECONDS * samplerate)
        self.delay_buffer_left = np.zeros(max_delay_samples, dtype=np.float32)
        self.delay_buffer_right = np.zeros(max_delay_samples, dtype=np.float32)
        self.delay_write_pos = 0
        
        # Reverb impulse response (pre-generated)
        self.reverb_ir = self._generate_reverb_ir(0.5, 0.5)
        
        # Filter states for bass boost and highpass
        self.bass_filter_state_left = None
        self.bass_filter_state_right = None
        self.highpass_filter_state_left = None
        self.highpass_filter_state_right = None
    
    def apply_effects(self, audio: np.ndarray, effects_config: Dict) -> np.ndarray:
        """Apply enabled effects to audio chunk
        
        Args:
            audio: Audio data (samples, channels) or (samples,) for mono
            effects_config: Dictionary of effect configurations
            
        Returns:
            Processed audio with same shape as input
        """
        if audio.size == 0:
            return audio
        
        # Ensure audio is 2D (samples, channels)
        is_mono = audio.ndim == 1
        if is_mono:
            audio = audio.reshape(-1, 1)
        
        # Make a copy to avoid modifying original
        processed = audio.copy()
        
        try:
            # Apply effects in order
            if effects_config.get('highpass', {}).get('enabled', False):
                cutoff = effects_config['highpass'].get('cutoff', 80)
                processed = self.apply_highpass(processed, cutoff)
            
            if effects_config.get('bassBoost', {}).get('enabled', False):
                gain_db = effects_config['bassBoost'].get('gain', 6)
                frequency = effects_config['bassBoost'].get('frequency', 100)
                processed = self.apply_bass_boost(processed, gain_db, frequency)
            
            if effects_config.get('distortion', {}).get('enabled', False):
                drive = effects_config['distortion'].get('drive', 0.5)
                processed = self.apply_distortion(processed, drive)
            
            if effects_config.get('echo', {}).get('enabled', False):
                delay_ms = effects_config['echo'].get('delay', 250)
                feedback = effects_config['echo'].get('feedback', 0.3)
                processed = self.apply_echo(processed, delay_ms, feedback)
            
            if effects_config.get('reverb', {}).get('enabled', False):
                room_size = effects_config['reverb'].get('roomSize', 0.5)
                damping = effects_config['reverb'].get('damping', 0.5)
                processed = self.apply_reverb(processed, room_size, damping)
            
        except Exception as e:
            logger.error(f"Error applying effects: {e}")
            return audio  # Return original on error
        
        # Convert back to mono if input was mono
        if is_mono:
            processed = processed.flatten()
        
        return processed
    
    def apply_reverb(self, audio: np.ndarray, room_size: float, damping: float) -> np.ndarray:
        """Apply reverb using convolution
        
        Args:
            audio: Audio data (samples, channels)
            room_size: Room size (0.0 - 1.0)
            damping: Damping factor (0.0 - 1.0)
            
        Returns:
            Audio with reverb applied
        """
        # Regenerate IR if parameters changed significantly
        self.reverb_ir = self._generate_reverb_ir(room_size, damping)
        
        # Apply convolution to each channel
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            # Use 'same' mode to keep same length
            reverbed = signal.fftconvolve(audio[:, ch], self.reverb_ir, mode='same')
            # Mix with dry signal (50% wet)
            result[:, ch] = 0.5 * audio[:, ch] + 0.5 * reverbed
        
        return result.astype(np.float32)
    
    def _generate_reverb_ir(self, room_size: float, damping: float) -> np.ndarray:
        """Generate reverb impulse response
        
        Args:
            room_size: Room size (0.0 - 1.0)
            damping: Damping factor (0.0 - 1.0)
            
        Returns:
            Impulse response array
        """
        # Scale IR length by room size
        ir_length = int(self.REVERB_IR_LENGTH * (0.3 + 0.7 * room_size))
        
        # Generate exponentially decaying noise
        decay_rate = 5.0 * (1.0 - damping)  # Higher damping = faster decay
        t = np.arange(ir_length) / self.samplerate
        envelope = np.exp(-decay_rate * t)
        
        # Random noise for diffusion
        noise = np.random.randn(ir_length)
        
        # Apply envelope
        ir = noise * envelope
        
        # Normalize
        ir = ir / np.max(np.abs(ir))
        
        return ir.astype(np.float32)
    
    def apply_echo(self, audio: np.ndarray, delay_ms: float, feedback: float) -> np.ndarray:
        """Apply echo/delay effect
        
        Args:
            audio: Audio data (samples, channels)
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0.0 - 0.9)
            
        Returns:
            Audio with echo applied
        """
        delay_samples = int((delay_ms / 1000.0) * self.samplerate)
        delay_samples = max(1, min(delay_samples, len(self.delay_buffer_left) - 1))
        
        feedback = max(0.0, min(0.9, feedback))  # Clamp to safe range
        
        result = audio.copy()
        
        # Process each sample
        for i in range(len(audio)):
            for ch in range(audio.shape[1]):
                # Select buffer for channel
                delay_buffer = self.delay_buffer_left if ch == 0 else self.delay_buffer_right
                
                # Read from delay buffer
                read_pos = (self.delay_write_pos - delay_samples) % len(delay_buffer)
                delayed_sample = delay_buffer[read_pos]
                
                # Mix with input
                output_sample = audio[i, ch] + feedback * delayed_sample
                
                # Write to delay buffer
                delay_buffer[self.delay_write_pos] = output_sample
                
                result[i, ch] = output_sample
            
            # Advance write position
            self.delay_write_pos = (self.delay_write_pos + 1) % len(self.delay_buffer_left)
        
        return result.astype(np.float32)
    
    def apply_bass_boost(self, audio: np.ndarray, gain_db: float, frequency: float) -> np.ndarray:
        """Apply bass boost using low-shelf filter
        
        Args:
            audio: Audio data (samples, channels)
            gain_db: Gain in decibels (0 - 12)
            frequency: Shelf frequency in Hz (50 - 200)
            
        Returns:
            Audio with bass boost applied
        """
        # Convert gain to linear
        gain_linear = 10 ** (gain_db / 20.0)
        
        # Design low-shelf filter
        # Using a simple first-order shelf
        freq_normalized = frequency / (self.samplerate / 2)
        freq_normalized = max(0.01, min(0.99, freq_normalized))
        
        # Butterworth low-shelf approximation
        b, a = signal.butter(2, freq_normalized, btype='low')
        
        # Apply gain to filter coefficients
        b = b * gain_linear
        
        # Apply filter to each channel with state preservation
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            if ch == 0:
                if self.bass_filter_state_left is None:
                    self.bass_filter_state_left = signal.lfilter_zi(b, a)
                
                filtered, self.bass_filter_state_left = signal.lfilter(
                    b, a, audio[:, ch], zi=self.bass_filter_state_left
                )
            else:
                if self.bass_filter_state_right is None:
                    self.bass_filter_state_right = signal.lfilter_zi(b, a)
                
                filtered, self.bass_filter_state_right = signal.lfilter(
                    b, a, audio[:, ch], zi=self.bass_filter_state_right
                )
            
            # Mix with dry signal to create shelf effect
            result[:, ch] = 0.5 * audio[:, ch] + 0.5 * filtered
        
        return result.astype(np.float32)
    
    def apply_highpass(self, audio: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """Apply high-pass filter
        
        Args:
            audio: Audio data (samples, channels)
            cutoff_freq: Cutoff frequency in Hz (20 - 500)
            
        Returns:
            Audio with high-pass filter applied
        """
        # Normalize frequency
        freq_normalized = cutoff_freq / (self.samplerate / 2)
        freq_normalized = max(0.01, min(0.99, freq_normalized))
        
        # Design Butterworth high-pass filter
        b, a = signal.butter(2, freq_normalized, btype='high')
        
        # Apply filter to each channel with state preservation
        result = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            if ch == 0:
                if self.highpass_filter_state_left is None:
                    self.highpass_filter_state_left = signal.lfilter_zi(b, a)
                
                filtered, self.highpass_filter_state_left = signal.lfilter(
                    b, a, audio[:, ch], zi=self.highpass_filter_state_left
                )
            else:
                if self.highpass_filter_state_right is None:
                    self.highpass_filter_state_right = signal.lfilter_zi(b, a)
                
                filtered, self.highpass_filter_state_right = signal.lfilter(
                    b, a, audio[:, ch], zi=self.highpass_filter_state_right
                )
            
            result[:, ch] = filtered
        
        return result.astype(np.float32)
    
    def apply_distortion(self, audio: np.ndarray, drive: float) -> np.ndarray:
        """Apply soft-clipping distortion
        
        Args:
            audio: Audio data (samples, channels)
            drive: Drive amount (0.0 - 1.0)
            
        Returns:
            Audio with distortion applied
        """
        # Scale drive to useful range
        gain = 1.0 + drive * 9.0  # 1x to 10x gain
        
        # Apply gain
        driven = audio * gain
        
        # Soft clipping using tanh
        distorted = np.tanh(driven)
        
        # Compensate for volume loss
        distorted = distorted * 0.8
        
        return distorted.astype(np.float32)
    
    def reset_state(self):
        """Reset all effect states (call when starting new playback)"""
        # Clear delay buffers
        self.delay_buffer_left.fill(0)
        self.delay_buffer_right.fill(0)
        self.delay_write_pos = 0
        
        # Reset filter states
        self.bass_filter_state_left = None
        self.bass_filter_state_right = None
        self.highpass_filter_state_left = None
        self.highpass_filter_state_right = None
