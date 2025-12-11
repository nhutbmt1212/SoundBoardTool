"""Backup configuration management"""
import json
from pathlib import Path
from .config_paths import get_config_path

# Backup configuration file
BACKUP_CONFIG_FILE = Path(get_config_path("backup_config.json"))

# Google Drive API scopes
GOOGLE_DRIVE_SCOPES = [
    'https://www.googleapis.com/auth/drive.file',  # Access to files created by the app
    'openid',  # OpenID Connect for authentication
    'https://www.googleapis.com/auth/userinfo.email',  # Access to user's email
    'https://www.googleapis.com/auth/userinfo.profile'  # Access to user's profile (name, picture)
]

# Backup folder name in Google Drive
BACKUP_FOLDER_NAME = "SoundboardPro_Backup"

# Files to backup
BACKUP_FILES = {
    'settings': 'sound_settings.json',
    'config': 'soundboard_config.json'
}


class BackupConfig:
    """Manages backup configuration and credentials"""
    
    def __init__(self):
        self.is_logged_in = False
        self.user_email = None
        self.user_name = None
        self.credentials_token = None
        self.last_backup_time = None
        self.auto_backup_enabled = False
        self._load()
    
    def _load(self):
        """Load backup configuration from file"""
        if BACKUP_CONFIG_FILE.exists():
            try:
                data = json.loads(BACKUP_CONFIG_FILE.read_text(encoding='utf-8'))
                self.is_logged_in = data.get('is_logged_in', False)
                self.user_email = data.get('user_email')
                self.user_name = data.get('user_name')
                self.credentials_token = data.get('credentials_token')
                self.last_backup_time = data.get('last_backup_time')
                self.auto_backup_enabled = data.get('auto_backup_enabled', False)
            except Exception:
                pass
    
    def save(self):
        """Save backup configuration to file"""
        try:
            data = {
                'is_logged_in': self.is_logged_in,
                'user_email': self.user_email,
                'user_name': self.user_name,
                'credentials_token': self.credentials_token,
                'last_backup_time': self.last_backup_time,
                'auto_backup_enabled': self.auto_backup_enabled
            }
            BACKUP_CONFIG_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception:
            pass
    
    def set_credentials(self, email: str, token: str, name: str = None):
        """Set user credentials after successful login"""
        self.is_logged_in = True
        self.user_email = email
        self.user_name = name
        self.credentials_token = token
        self.save()
    
    def clear_credentials(self):
        """Clear user credentials (logout)"""
        self.is_logged_in = False
        self.user_email = None
        self.user_name = None
        self.credentials_token = None
        self.auto_backup_enabled = False
        self.save()
    
    def update_last_backup_time(self, timestamp: str):
        """Update last backup timestamp"""
        self.last_backup_time = timestamp
        self.save()
    
    def set_auto_backup(self, enabled: bool):
        """Enable or disable auto backup"""
        self.auto_backup_enabled = enabled
        self.save()
