"""Microphone Passthrough - Routes mic audio to VB-Cable with minimal latency"""
import numpy as np
import threading

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    sd = None

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class MicPassthrough:
    """Handles microphone audio routing to VB-Cable with optimized low-latency passthrough"""
    
    def __init__(self, vb_manager):
        self.vb_manager = vb_manager
        self.device_id = None
        self.volume = 1.0
        self.enabled = False
        
        self._input_stream = None
        self._output_stream = None
        self._buffer = None
        self._buffer_lock = threading.Lock()
        
        # Noise gate parameters - tuned to eliminate feedback when silent
        self._noise_gate_threshold = 0.04  # -28dB threshold (blocks feedback, allows voice)
        self._noise_gate_ratio = 0.01   # Fade to 1% when below threshold (aggressive suppression)
        self._gate_smoothing = 0.92        # Smooth gate transitions (faster response)
        self._current_gate = 1.0           # Current gate level
        
        self._detect_mic()
    
    def _detect_mic(self):
        """Auto-detect default microphone"""
        if not SD_AVAILABLE:
            return
        
        try:
            default_input = sd.default.device[0]
            if default_input is not None and default_input >= 0:
                dev = sd.query_devices(default_input)
                if dev['max_input_channels'] > 0:
                    self.device_id = default_input
                    print(f"✓ Mic: {dev['name']}")
        except Exception:
            pass
    
    def get_devices(self) -> list:
        """Get list of available microphones"""
        if not SD_AVAILABLE:
            return []
        
        devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    devices.append({'id': i, 'name': dev['name']})
        except Exception:
            pass
        return devices
    
    def set_device(self, device_id: int):
        """Set microphone device"""
        was_enabled = self.enabled
        if was_enabled:
            self.stop()
        self.device_id = device_id
        if was_enabled:
            self.start()
    
    def set_volume(self, vol: float):
        """Set passthrough volume (0.0 - 3.0)"""
        self.volume = max(0.0, min(3.0, vol))
        print(f"Mic volume set to: {self.volume:.2f}")
    
    def _apply_noise_gate(self, data):
        """Apply noise gate to reduce background noise"""
        # Calculate RMS (root mean square) for volume detection
        rms = np.sqrt(np.mean(data ** 2))
        
        # Determine target gate level
        if rms > self._noise_gate_threshold:
            target_gate = 1.0  # Full volume when speaking
        else:
            target_gate = self._noise_gate_ratio  # Reduce to 5% when silent
        
        # Smooth gate transitions to avoid clicking
        self._current_gate = (self._gate_smoothing * self._current_gate + 
                             (1 - self._gate_smoothing) * target_gate)
        
        # Apply gate
        return data * self._current_gate
    
    def start(self) -> bool:
        """Start routing mic to VB-Cable with minimal latency"""
        if not SD_AVAILABLE:
            return False
        
        if self.device_id is None or not self.vb_manager.is_connected():
            print(f"Cannot start mic: mic={self.device_id}, vb={self.vb_manager.is_connected()}")
            return False
        
        if self._input_stream is not None:
            return True  # Already running
        
        try:
            # Get sample rates
            vb_samplerate = self.vb_manager.get_samplerate()
            mic_info = sd.query_devices(self.device_id)
            mic_samplerate = int(mic_info['default_samplerate'])
            
            # Optimal blocksize for balance between latency and stability
            blocksize = 512
            
            # Buffer size: 10 frames for stability without too much latency
            # At 48kHz with blocksize=512: ~106ms total buffer (acceptable for voice)
            buffer_frames = 10
            self._buffer = np.zeros((blocksize * buffer_frames, 1), dtype='float32')
            self._write_pos = 0
            self._read_pos = 0
            
            # Pre-fill buffer to middle to prevent initial underrun
            self._write_pos = (blocksize * buffer_frames) // 2
            
            # Reset noise gate
            self._current_gate = 1.0
            
            print(f"Mic passthrough: mic={mic_samplerate}Hz, vb={vb_samplerate}Hz, blocksize={blocksize}")
            print(f"Noise gate: enabled (threshold={self._noise_gate_threshold:.3f})")
            
            # Store reference to self for callbacks
            engine = self
            
            def input_callback(indata, frames, time, status):
                """Capture mic input with volume applied"""
                if status and 'overflow' not in str(status).lower():
                    print(f"Mic input: {status}")
                
                # Apply volume and write to circular buffer
                with engine._buffer_lock:
                    # First apply noise gate to reduce background noise
                    data = engine._apply_noise_gate(indata)
                    
                    # Then apply volume (so volume slider works)
                    data = data * engine.volume
                    
                    # Handle sample rate mismatch with high-quality resampling
                    if mic_samplerate != vb_samplerate:
                        if SCIPY_AVAILABLE:
                            # Use scipy for high-quality resampling
                            ratio = vb_samplerate / mic_samplerate
                            new_length = int(len(data) * ratio)
                            data = signal.resample(data, new_length).astype('float32')
                        else:
                            # Fallback to linear interpolation
                            ratio = vb_samplerate / mic_samplerate
                            new_length = int(len(data) * ratio)
                            indices = np.linspace(0, len(data) - 1, new_length)
                            data = np.interp(indices, np.arange(len(data)), data.flatten()).reshape(-1, 1).astype('float32')
                    
                    # Write to circular buffer
                    buffer_len = len(engine._buffer)
                    data_len = len(data)
                    
                    if engine._write_pos + data_len <= buffer_len:
                        engine._buffer[engine._write_pos:engine._write_pos + data_len] = data
                    else:
                        # Wrap around
                        first_part = buffer_len - engine._write_pos
                        engine._buffer[engine._write_pos:] = data[:first_part]
                        engine._buffer[:data_len - first_part] = data[first_part:]
                    
                    engine._write_pos = (engine._write_pos + data_len) % buffer_len
            
            def output_callback(outdata, frames, time, status):
                """Output to VB-Cable"""
                if status and 'underflow' not in str(status).lower():
                    print(f"VB output: {status}")
                
                with engine._buffer_lock:
                    buffer_len = len(engine._buffer)
                    data_len = len(outdata)
                    
                    # Check if we have enough data
                    available = (engine._write_pos - engine._read_pos) % buffer_len
                    if available < data_len:
                        # Not enough data, output silence to prevent glitches
                        outdata[:] = 0
                        return
                    
                    # Read from circular buffer
                    if engine._read_pos + data_len <= buffer_len:
                        outdata[:] = engine._buffer[engine._read_pos:engine._read_pos + data_len]
                    else:
                        # Wrap around
                        first_part = buffer_len - engine._read_pos
                        outdata[:first_part] = engine._buffer[engine._read_pos:]
                        outdata[first_part:] = engine._buffer[:data_len - first_part]
                    
                    engine._read_pos = (engine._read_pos + data_len) % buffer_len
            
            # Create input stream
            self._input_stream = sd.InputStream(
                device=self.device_id,
                samplerate=mic_samplerate,
                channels=1,
                dtype='float32',
                callback=input_callback,
                blocksize=blocksize,
                latency='low'
            )
            
            # Create output stream
            self._output_stream = sd.OutputStream(
                device=self.vb_manager.device_id,
                samplerate=vb_samplerate,
                channels=1,
                dtype='float32',
                callback=output_callback,
                blocksize=blocksize,
                latency='low'
            )
            
            self._input_stream.start()
            self._output_stream.start()
            self.enabled = True
            
            print(f"✓ Mic passthrough started (low latency mode)")
            return True
            
        except Exception as e:
            print(f"Mic passthrough error: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
            return False
    
    def stop(self):
        """Stop mic passthrough"""
        if self._input_stream is not None:
            try:
                self._input_stream.abort()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        
        if self._output_stream is not None:
            try:
                self._output_stream.abort()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None
        
        self._buffer = None
        self.enabled = False
        print("Mic passthrough stopped")
    
    def is_enabled(self) -> bool:
        return self.enabled
