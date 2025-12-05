"""
Configuration settings for Soundboard - Lưu và load từ file
"""
import json
import os
from pathlib import Path


class Config:
    CONFIG_FILE = "soundboard_config.json"
    
    def __init__(self):
        self.sounds_dir = "sounds"
        self.default_volume = 0.7
        self.max_buttons_per_row = 4
        
        # Audio routing settings
        self.routing_enabled = False
        self.routing_device_index = None
        self.routing_device_name = ""
        
        # Load saved config
        self.load()
    
    def get_config_path(self):
        """Get config file path"""
        # Lưu trong thư mục user hoặc cùng thư mục app
        app_dir = Path(__file__).parent.parent
        return app_dir / self.CONFIG_FILE
    
    def load(self):
        """Load config from file"""
        config_path = self.get_config_path()
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.sounds_dir = data.get('sounds_dir', self.sounds_dir)
                self.default_volume = data.get('default_volume', self.default_volume)
                self.max_buttons_per_row = data.get('max_buttons_per_row', self.max_buttons_per_row)
                self.routing_enabled = data.get('routing_enabled', False)
                self.routing_device_index = data.get('routing_device_index', None)
                self.routing_device_name = data.get('routing_device_name', "")
                
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save(self):
        """Save config to file"""
        config_path = self.get_config_path()
        
        data = {
            'sounds_dir': self.sounds_dir,
            'default_volume': self.default_volume,
            'max_buttons_per_row': self.max_buttons_per_row,
            'routing_enabled': self.routing_enabled,
            'routing_device_index': self.routing_device_index,
            'routing_device_name': self.routing_device_name,
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def set_routing(self, enabled, device_index=None, device_name=""):
        """Set routing config and save"""
        self.routing_enabled = enabled
        self.routing_device_index = device_index
        self.routing_device_name = device_name
        self.save()
