"""
Audio Router - Route soundboard output to virtual audio device
"""
import pyaudio
import numpy as np
from threading import Thread
import queue

class AudioRouter:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.is_routing = False
        self.audio_queue = queue.Queue()
        self.output_device_index = None
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 1024
        
    def list_audio_devices(self):
        """List all available audio devices"""
        devices = []
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'max_output_channels': info['maxOutputChannels'],
                'max_input_channels': info['maxInputChannels'],
                'default_sample_rate': info['defaultSampleRate']
            })
        return devices
    
    def get_virtual_devices(self):
        """Get list of virtual audio devices (VB-Cable, NVIDIA, etc)"""
        devices = self.list_audio_devices()
        virtual_devices = []
        
        for device in devices:
            name_lower = device['name'].lower()
            
            # Chỉ lấy OUTPUT devices (max_output_channels > 0)
            if device['max_output_channels'] > 0:
                # VB-Cable: "Speakers (VB-Audio Virtual Cable)" hoặc "CABLE Input"
                if 'vb-audio' in name_lower:
                    virtual_devices.append(device)
                # Voicemeeter
                elif 'voicemeeter' in name_lower:
                    virtual_devices.append(device)
                # NVIDIA Virtual Audio
                elif 'nvidia' in name_lower and 'virtual' in name_lower:
                    virtual_devices.append(device)
        
        return virtual_devices
    
    def get_all_output_devices(self):
        """Get ALL output devices (để user có thể chọn bất kỳ device nào)"""
        devices = self.list_audio_devices()
        output_devices = []
        
        for device in devices:
            if device['max_output_channels'] > 0:
                output_devices.append(device)
        
        return output_devices
    
    def set_output_device(self, device_index):
        """Set the output device for routing"""
        self.output_device_index = device_index
    
    def start_routing(self):
        """Start routing audio to virtual device"""
        if self.output_device_index is None:
            return False
        
        self.is_routing = True
        self.routing_thread = Thread(target=self._routing_loop, daemon=True)
        self.routing_thread.start()
        return True
    
    def stop_routing(self):
        """Stop routing audio"""
        self.is_routing = False
    
    def _routing_loop(self):
        """Main routing loop"""
        try:
            stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_routing:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    stream.write(audio_data.tobytes())
                except queue.Empty:
                    # Send silence to keep stream alive
                    silence = np.zeros((self.chunk_size, self.channels), dtype=np.float32)
                    stream.write(silence.tobytes())
            
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Routing error: {e}")
    
    def send_audio(self, audio_data):
        """Send audio data to routing queue"""
        if self.is_routing:
            self.audio_queue.put(audio_data)
    
    def cleanup(self):
        """Cleanup audio resources"""
        self.stop_routing()
        self.p.terminate()
