"""Backup API endpoints for Google Drive integration"""
import eel
import json
from services.google_drive_service import GoogleDriveService

# Global service instance
_drive_service = None


def get_drive_service() -> GoogleDriveService:
    """Get or create Google Drive service instance"""
    global _drive_service
    if _drive_service is None:
        _drive_service = GoogleDriveService()
    return _drive_service


@eel.expose
def get_backup_status():
    """Get current backup status
    
    Returns:
        dict: Backup status including login state, user email, last backup time
    """
    try:
        service = get_drive_service()
        return service.get_status()
    except Exception as e:
        print(f"[BackupAPI] Failed to get status: {e}")
        return {
            'is_logged_in': False,
            'user_email': None,
            'user_name': None,
            'last_backup_time': None,
            'auto_backup_enabled': False
        }


@eel.expose
def start_google_login():
    """Start Google OAuth login flow with automatic callback handling
    
    Returns:
        dict: Contains 'success' and 'email' or 'error'
    """
    try:
        import webbrowser
        import threading
        from services import oauth_callback_server
        
        service = get_drive_service()
        
        # Start OAuth flow
        auth_url = service.start_oauth_flow()
        if not auth_url:
            return {
                'success': False,
                'error': 'Failed to start OAuth flow'
            }
        
        print(f"[BackupAPI] Starting OAuth callback server...")
        
        # Start callback server
        oauth_callback_server.start_callback_server()
        
        # Open browser
        print(f"[BackupAPI] Opening browser for authorization...")
        webbrowser.open(auth_url)
        
        # Wait for callback in background
        def wait_and_complete():
            print(f"[BackupAPI] Waiting for authorization...")
            
            # Wait for callback (2 minutes timeout)
            if oauth_callback_server.wait_for_callback(timeout=120):
                # Get authorization response
                auth_response = oauth_callback_server.get_authorization_response()
                
                if auth_response:
                    print(f"[BackupAPI] Completing OAuth flow...")
                    result = service.complete_oauth_flow(auth_response)
                    
                    # Store result globally so frontend can poll it
                    global _login_result
                    _login_result = result
                    print(f"[BackupAPI] Stored login result: {result}")
                    
                    if result.get('success'):
                        print(f"[BackupAPI] ✅ Login successful: {result.get('email')}")
                    else:
                        print(f"[BackupAPI] ❌ Login failed: {result.get('error')}")
                else:
                    print(f"[BackupAPI] ❌ No authorization response received")
                    _login_result = {'success': False, 'error': 'No authorization response'}
            else:
                print(f"[BackupAPI] ❌ Authorization timeout")
                _login_result = {'success': False, 'error': 'Authorization timeout'}
            
            # Stop callback server
            oauth_callback_server.stop_callback_server()
        
        # Start waiting in background
        threading.Thread(target=wait_and_complete, daemon=True).start()
        
        return {
            'success': True,
            'message': 'Browser opened, waiting for authorization...'
        }
        
    except Exception as e:
        print(f"[BackupAPI] Failed to start login: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


# Global variable to store login result
_login_result = None


@eel.expose
def check_login_status():
    """Check if login has completed
    
    Returns:
        dict: Login result or None if still waiting
    """
    global _login_result
    result = _login_result
    print(f"[BackupAPI] check_login_status called, returning: {result}")
    if result:
        _login_result = None  # Clear after reading
    return result


@eel.expose
def complete_google_login(authorization_response: str):
    """Complete Google OAuth login flow (legacy - for manual flow)
    
    Args:
        authorization_response: Full authorization response URL
        
    Returns:
        dict: Contains 'success' and 'email' or 'error'
    """
    try:
        service = get_drive_service()
        result = service.complete_oauth_flow(authorization_response)
        return result
    except Exception as e:
        print(f"[BackupAPI] Failed to complete login: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@eel.expose
def google_logout():
    """Logout from Google Drive
    
    Returns:
        dict: Contains 'success'
    """
    try:
        service = get_drive_service()
        service.logout()
        return {'success': True}
    except Exception as e:
        print(f"[BackupAPI] Failed to logout: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@eel.expose
def backup_to_drive():
    """Backup all settings to Google Drive
    
    Returns:
        dict: Backup result with files backed up/failed
    """
    try:
        service = get_drive_service()
        
        if not service.config.is_logged_in:
            return {
                'success': False,
                'error': 'Not logged in',
                'require_login': True
            }
        
        result = service.backup_all_settings()
        return result
    except Exception as e:
        print(f"[BackupAPI] Failed to backup: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@eel.expose
def restore_from_drive():
    """Restore all settings from Google Drive
    
    Returns:
        dict: Restore result with files restored/failed
    """
    try:
        service = get_drive_service()
        
        if not service.config.is_logged_in:
            return {
                'success': False,
                'error': 'Not logged in',
                'require_login': True
            }
        
        result = service.restore_all_settings()
        return result
    except Exception as e:
        print(f"[BackupAPI] Failed to restore: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@eel.expose
def list_drive_backups():
    """List all available backups from Google Drive
    
    Returns:
        dict: Contains 'success' and 'backups' list or 'error'
    """
    try:
        service = get_drive_service()
        
        if not service.config.is_logged_in:
            return {
                'success': False,
                'error': 'Not logged in',
                'require_login': True
            }
        
        backups = service.list_backups()
        return {
            'success': True,
            'backups': backups
        }
    except Exception as e:
        print(f"[BackupAPI] Failed to list backups: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@eel.expose
def set_auto_backup(enabled: bool):
    """Enable or disable auto backup
    
    Args:
        enabled: True to enable auto backup, False to disable
        
    Returns:
        dict: Contains 'success'
    """
    try:
        service = get_drive_service()
        service.set_auto_backup(enabled)
        return {'success': True}
    except Exception as e:
        print(f"[BackupAPI] Failed to set auto backup: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def trigger_auto_backup():
    """Trigger auto backup if enabled and logged in
    
    This function should be called after settings changes
    """
    try:
        service = get_drive_service()
        
        if service.config.is_logged_in and service.config.auto_backup_enabled:
            print("[BackupAPI] Auto backup triggered")
            service.backup_all_settings()
    except Exception as e:
        print(f"[BackupAPI] Auto backup failed: {e}")
