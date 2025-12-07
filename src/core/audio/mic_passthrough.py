"""Microphone Passthrough - Routes mic audio to VB-Cable"""
import queue

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    sd = None


class MicPassthrough:
    """Handles microphone audio routing to VB-Cable"""
    
    def __init__(self, vb_manager):
        self.vb_manager = vb_manager
        self.device_id = None
        self.volume = 1.0
        self.enabled = False
        
        self._input_stream = None
        self._output_stream = None
        self._buffer = None
        self._resample_ratio = None
        
        # Overflow tracking
        self._overflow_count = 0
        self._last_overflow_report = 0
        
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
        """Set passthrough volume (0.0 - 2.0)"""
        self.volume = max(0.0, min(2.0, vol))
    
    def start(self) -> bool:
        """Start routing mic to VB-Cable"""
        if not SD_AVAILABLE:
            return False
        
        if self.device_id is None or not self.vb_manager.is_connected():
            print(f"Cannot start mic: mic={self.device_id}, vb={self.vb_manager.is_connected()}")
            return False
        
        if self._input_stream is not None:
            return True  # Already running
        
        try:
            # Larger buffer to prevent overflow (was 20, now 100)
            self._buffer = queue.Queue(maxsize=100)
            self._overflow_count = 0
            self._last_overflow_report = 0
            
            # Get sample rates
            vb_samplerate = self.vb_manager.get_samplerate()
            mic_info = sd.query_devices(self.device_id)
            mic_samplerate = int(mic_info['default_samplerate'])
            
            blocksize = 512
            self._resample_ratio = mic_samplerate / vb_samplerate if mic_samplerate != vb_samplerate else None
            
            print(f"Mic passthrough: mic={mic_samplerate}Hz, vb={vb_samplerate}Hz")
            
            # Create callbacks
            engine = self
            
            def input_callback(indata, frames, time, status):
                # Track overflow but don't spam console
                if status:
                    if 'overflow' in str(status).lower():
                        engine._overflow_count += 1
                        # Report every 100 overflows instead of every single one
                        if engine._overflow_count - engine._last_overflow_report >= 100:
                            print(f"Mic input overflow (total: {engine._overflow_count})")
                            engine._last_overflow_report = engine._overflow_count
                    else:
                        print(f"Mic input: {status}")
                
                try:
                    data = indata.copy() * engine.volume
                    
                    if engine._resample_ratio and engine._resample_ratio != 1.0:
                        from scipy import signal
                        target_length = int(len(data) / engine._resample_ratio)
                        data = signal.resample(data, target_length)
                    
                    engine._buffer.put_nowait(data)
                except queue.Full:
                    # Buffer full, skip this frame (graceful degradation)
                    pass
                except Exception as e:
                    print(f"Mic resample error: {e}")
            
            def output_callback(outdata, frames, time, status):
                if status:
                    print(f"VB output: {status}")
                try:
                    data = engine._buffer.get_nowait()
                    if len(data) >= len(outdata):
                        outdata[:] = data[:len(outdata)]
                    else:
                        outdata[:len(data)] = data
                        outdata[len(data):] = 0
                except queue.Empty:
                    outdata[:] = 0
            
            # Create streams
            self._input_stream = sd.InputStream(
                device=self.device_id,
                samplerate=mic_samplerate,
                channels=1,
                dtype='float32',
                callback=input_callback,
                blocksize=blocksize,
            )
            
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
            
            print(f"✓ Mic passthrough started")
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
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        
        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None
        
        self._buffer = None
        self.enabled = False
    
    def is_enabled(self) -> bool:
        return self.enabled
