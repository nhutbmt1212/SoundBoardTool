"""VB-Cable Device Manager - Detection and management"""
import logging

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    sd = None

# Constants
DEFAULT_SAMPLERATE = 48000  # Standard audio sample rate
PREFERRED_CHANNELS = 2      # Stereo output


class VBCableManager:
    """Manages VB-Cable virtual audio device detection and configuration"""
    
    def __init__(self):
        self.device_id = None
        self.enabled = False
        self.samplerate = None
        self._detect()
    
    def _detect(self):
        """Auto-detect VB-Cable device - prefer 2-channel MME/DirectSound"""
        if not SD_AVAILABLE:
            return
        
        try:
            vb_devices = []
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_output_channels'] > 0:
                    name_lower = dev['name'].lower()
                    if 'vb-audio virtual cable' in name_lower and 'speakers' in name_lower:
                        vb_devices.append((i, dev))
            
            if not vb_devices:
                return
            
            # Prefer device with exactly 2 channels (MME/DirectSound) - more stable
            for i, dev in vb_devices:
                if dev['max_output_channels'] == PREFERRED_CHANNELS:
                    self._set_device(i, dev)
                    return
            
            # Fallback to first device
            i, dev = vb_devices[0]
            self._set_device(i, dev)
            
        except Exception as e:
            logger.error(f"VB-Cable detection error: {e}")
    
    def _set_device(self, device_id: int, dev: dict):
        """Set the VB-Cable device"""
        self.device_id = device_id
        self.enabled = True
        print(f"✓ VB-Cable: [{device_id}] {dev['name']} ({dev['max_output_channels']} ch)")
        
        # Cache sample rate
        try:
            self.samplerate = int(dev['default_samplerate'])
        except Exception as e:
            logger.debug(f"Failed to get sample rate: {e}, using default")
            self.samplerate = DEFAULT_SAMPLERATE
    
    def is_connected(self) -> bool:
        """Check if VB-Cable is available"""
        return self.enabled and self.device_id is not None
    
    def get_device_info(self) -> dict:
        """Get VB-Cable device info"""
        if not self.is_connected() or not SD_AVAILABLE:
            return {}
        
        try:
            return sd.query_devices(self.device_id)
        except Exception:
            return {}
    
    def get_samplerate(self) -> int:
        """Get VB-Cable sample rate"""
        return self.samplerate or DEFAULT_SAMPLERATE
    
    def get_channels(self) -> int:
        """Get VB-Cable channel count"""
        info = self.get_device_info()
        return min(PREFERRED_CHANNELS, info.get('max_output_channels', PREFERRED_CHANNELS))
