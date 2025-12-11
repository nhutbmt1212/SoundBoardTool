"""Google Drive backup service"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# Allow insecure transport for localhost (required for desktop OAuth)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

from core.backup_config import BackupConfig, GOOGLE_DRIVE_SCOPES, BACKUP_FOLDER_NAME
from core.config_paths import get_config_path, get_sounds_dir


class GoogleDriveService:
    """Handles Google Drive backup and restore operations"""
    
    # OAuth 2.0 Client configuration
    CLIENT_CONFIG = {
        "installed": {
            "client_id": "1032504429798-tgsqa0nugauhu80dqaffp1hde4gujk1h.apps.googleusercontent.com",
            "project_id": "soundboardpro-backup",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "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",
            "redirect_uris": ["http://localhost:8080"]
        }
    }
    
    def __init__(self):
        self.config = BackupConfig()
        self.service = None
        self.backup_folder_id = None
        
        # Initialize service if already logged in
        if self.config.is_logged_in and self.config.credentials_token:
            self._initialize_service()
    
    def _initialize_service(self) -> bool:
        """Initialize Google Drive service with stored credentials"""
        try:
            if not self.config.credentials_token:
                return False
            
            # Reconstruct credentials from token
            creds = Credentials.from_authorized_user_info(
                json.loads(self.config.credentials_token),
                GOOGLE_DRIVE_SCOPES
            )
            
            # Build service
            self.service = build('drive', 'v3', credentials=creds)
            
            # Get or create backup folder
            self.backup_folder_id = self._get_or_create_backup_folder()
            
            return True
        except Exception as e:
            print(f"[GoogleDrive] Failed to initialize service: {e}")
            return False
    
    def start_oauth_flow(self) -> str:
        """Start OAuth flow and return authorization URL"""
        try:
            flow = Flow.from_client_config(
                self.CLIENT_CONFIG,
                scopes=GOOGLE_DRIVE_SCOPES,
                redirect_uri='http://localhost:8080'
            )
            
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            # Store flow for later use
            self._flow = flow
            
            return auth_url
        except Exception as e:
            print(f"[GoogleDrive] Failed to start OAuth flow: {e}")
            return None
    
    def complete_oauth_flow(self, authorization_response: str) -> Dict:
        """Complete OAuth flow with authorization response"""
        try:
            if not hasattr(self, '_flow'):
                return {'success': False, 'error': 'OAuth flow not started'}
            
            # Fetch token
            self._flow.fetch_token(authorization_response=authorization_response)
            
            # Get credentials
            creds = self._flow.credentials
            
            # Get email and name from ID token (no API call needed)
            email = None
            name = None
            if hasattr(creds, 'id_token') and creds.id_token:
                # ID token contains email and name claims
                import jwt
                try:
                    decoded = jwt.decode(creds.id_token, options={"verify_signature": False})
                    email = decoded.get('email', 'unknown@gmail.com')
                    name = decoded.get('name', None)  # Full name from Google account
                    print(f"[GoogleDrive] User info from ID token: {name} ({email})")
                except Exception as e:
                    print(f"[GoogleDrive] Failed to decode ID token: {e}")
                    email = 'unknown@gmail.com'
            else:
                # Fallback: use a placeholder
                email = 'user@gmail.com'
            
            # Save credentials
            token_json = creds.to_json()
            self.config.set_credentials(email, token_json, name)
            
            # Initialize service
            self._initialize_service()
            
            return {
                'success': True,
                'email': email,
                'name': name
            }
        except Exception as e:
            print(f"[GoogleDrive] Failed to complete OAuth: {e}")
            return {'success': False, 'error': str(e)}
    
    def logout(self):
        """Logout and clear credentials"""
        self.config.clear_credentials()
        self.service = None
        self.backup_folder_id = None
    
    def _get_or_create_backup_folder(self) -> Optional[str]:
        """Get or create backup folder in Google Drive"""
        try:
            # Search for existing folder
            query = f"name='{BACKUP_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()
            
            folders = results.get('files', [])
            
            if folders:
                return folders[0]['id']
            
            # Create new folder
            file_metadata = {
                'name': BACKUP_FOLDER_NAME,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = self.service.files().create(
                body=file_metadata,
                fields='id'
            ).execute()
            
            return folder.get('id')
        except Exception as e:
            print(f"[GoogleDrive] Failed to get/create backup folder: {e}")
            return None
    
    def backup_file(self, local_path: Path, drive_filename: str) -> bool:
        """Backup a single file to Google Drive"""
        try:
            if not self.service or not self.backup_folder_id:
                print(f"[GoogleDrive] Cannot backup: service or folder not initialized")
                return False
            
            if not local_path.exists():
                print(f"[GoogleDrive] File not found: {local_path}")
                return False
            
            print(f"[GoogleDrive] Uploading {drive_filename}...")
            
            # Check if file already exists in Drive
            query = f"name='{drive_filename}' and '{self.backup_folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id)'
            ).execute()
            
            existing_files = results.get('files', [])
            
            # Prepare file metadata
            file_metadata = {
                'name': drive_filename,
                'parents': [self.backup_folder_id]
            }
            
            media = MediaFileUpload(
                str(local_path),
                resumable=True
            )
            
            if existing_files:
                # Update existing file
                file_id = existing_files[0]['id']
                print(f"[GoogleDrive] Updating existing file (ID: {file_id})...")
                result = self.service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                print(f"[GoogleDrive] ✅ Updated: {drive_filename}")
            else:
                # Create new file
                print(f"[GoogleDrive] Creating new file...")
                result = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"[GoogleDrive] ✅ Created: {drive_filename} (ID: {result.get('id')})")
            
            return True
        except Exception as e:
            print(f"[GoogleDrive] Failed to backup file {drive_filename}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def backup_all_settings(self) -> Dict:
        """Backup entire AppData folder as ZIP archive"""
        try:
            if not self.service:
                return {'success': False, 'error': 'Not logged in'}
            
            from core.config_paths import get_app_data_dir
            import zipfile
            import tempfile
            
            app_data_dir = Path(get_app_data_dir())
            
            # Create temporary ZIP file
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False) as tmp_file:
                zip_path = Path(tmp_file.name)
            
            try:
                # Create ZIP archive of entire AppData folder
                print(f"[GoogleDrive] Creating backup archive...")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Walk through all files in AppData
                    for root, dirs, files in os.walk(app_data_dir):
                        # Skip backup_config.json (contains credentials)
                        for file in files:
                            if file == 'backup_config.json':
                                continue
                            
                            file_path = Path(root) / file
                            # Store relative path in ZIP
                            arcname = file_path.relative_to(app_data_dir)
                            zipf.write(file_path, arcname)
                            print(f"[GoogleDrive] Added: {arcname}")
                
                # Upload ZIP to Google Drive
                zip_filename = f"SoundboardPro_Backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                print(f"[GoogleDrive] Uploading {zip_filename}...")
                
                if self.backup_file(zip_path, zip_filename):
                    # Update last backup time
                    timestamp = datetime.now().isoformat()
                    self.config.update_last_backup_time(timestamp)
                    
                    return {
                        'success': True,
                        'filename': zip_filename,
                        'timestamp': timestamp,
                        'size': zip_path.stat().st_size
                    }
                else:
                    return {'success': False, 'error': 'Failed to upload backup'}
            
            finally:
                # Clean up temp file
                if zip_path.exists():
                    zip_path.unlink()
        
        except Exception as e:
            print(f"[GoogleDrive] Failed to backup: {e}")
            return {'success': False, 'error': str(e)}
    
    def restore_file(self, drive_filename: str, local_path: Path) -> bool:
        """Restore a single file from Google Drive"""
        try:
            if not self.service or not self.backup_folder_id:
                return False
            
            # Find file in Drive
            query = f"name='{drive_filename}' and '{self.backup_folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id)'
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                return False
            
            file_id = files[0]['id']
            
            # Download file
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            # Write to local file
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(fh.getvalue())
            
            return True
        except Exception as e:
            print(f"[GoogleDrive] Failed to restore file {drive_filename}: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List all available backups from Google Drive"""
        try:
            if not self.service or not self.backup_folder_id:
                return []
            
            # Search for all ZIP backups
            query = f"name contains 'SoundboardPro_Backup_' and name contains '.zip' and '{self.backup_folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name, createdTime, size)',
                orderBy='createdTime desc'
            ).execute()
            
            backups = results.get('files', [])
            return backups
        except Exception as e:
            print(f"[GoogleDrive] Failed to list backups: {e}")
            return []
    
    def restore_all_settings(self, backup_filename: Optional[str] = None) -> Dict:
        """Restore from ZIP backup (latest if filename not specified)"""
        try:
            if not self.service:
                return {'success': False, 'error': 'Not logged in'}
            
            from core.config_paths import get_app_data_dir
            import zipfile
            import tempfile
            
            # Get backup to restore
            if not backup_filename:
                # Get latest backup
                backups = self.list_backups()
                if not backups:
                    return {'success': False, 'error': 'No backups found'}
                backup_filename = backups[0]['name']
            
            print(f"[GoogleDrive] Restoring from {backup_filename}...")
            
            # Download ZIP to temp file
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.zip', delete=False) as tmp_file:
                zip_path = Path(tmp_file.name)
            
            try:
                if not self.restore_file(backup_filename, zip_path):
                    return {'success': False, 'error': 'Failed to download backup'}
                
                # Extract ZIP to AppData
                app_data_dir = Path(get_app_data_dir())
                print(f"[GoogleDrive] Extracting to {app_data_dir}...")
                
                files_restored = []
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    for file_info in zipf.filelist:
                        # Extract file
                        zipf.extract(file_info, app_data_dir)
                        files_restored.append(file_info.filename)
                        print(f"[GoogleDrive] Restored: {file_info.filename}")
                
                return {
                    'success': True,
                    'backup_filename': backup_filename,
                    'files_restored': files_restored,
                    'count': len(files_restored)
                }
            
            finally:
                # Clean up temp file
                if zip_path.exists():
                    zip_path.unlink()
        
        except Exception as e:
            print(f"[GoogleDrive] Failed to restore: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict:
        """Get current backup status"""
        return {
            'is_logged_in': self.config.is_logged_in,
            'user_email': self.config.user_email,
            'user_name': self.config.user_name,
            'last_backup_time': self.config.last_backup_time,
            'auto_backup_enabled': self.config.auto_backup_enabled
        }
    
    def set_auto_backup(self, enabled: bool):
        """Enable or disable auto backup"""
        self.config.set_auto_backup(enabled)
