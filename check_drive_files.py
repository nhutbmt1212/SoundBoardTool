"""Check what files are in Google Drive"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.google_drive_service import GoogleDriveService

def check_drive():
    """Check Google Drive contents"""
    print("=" * 60)
    print("Google Drive Contents Check")
    print("=" * 60)
    
    service = GoogleDriveService()
    
    # Check login
    status = service.get_status()
    if not status['is_logged_in']:
        print("\n❌ Not logged in!")
        return
    
    print(f"\n✅ Logged in as: {status['user_email']}")
    
    # Check if service is initialized
    if not service.service:
        print("\n❌ Service not initialized!")
        return
    
    print(f"\n[1] Backup folder ID: {service.backup_folder_id}")
    
    # List ALL files in Drive (not just in backup folder)
    print("\n[2] All files in your Google Drive:")
    try:
        results = service.service.files().list(
            pageSize=100,
            fields="files(id, name, mimeType, size, createdTime, parents)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print("   No files found!")
        else:
            print(f"   Found {len(files)} file(s):")
            for f in files:
                size = int(f.get('size', 0)) if f.get('size') else 0
                size_mb = size / 1024 / 1024 if size > 0 else 0
                mime = f.get('mimeType', 'unknown')
                parents = f.get('parents', [])
                
                print(f"\n   📄 {f['name']}")
                print(f"      ID: {f['id']}")
                print(f"      Type: {mime}")
                if size > 0:
                    print(f"      Size: {size_mb:.2f} MB")
                print(f"      Created: {f.get('createdTime', 'Unknown')}")
                if parents:
                    print(f"      Parents: {parents}")
    
    except Exception as e:
        print(f"\n❌ Error listing files: {e}")
        import traceback
        traceback.print_exc()
    
    # List files in backup folder specifically
    if service.backup_folder_id:
        print(f"\n[3] Files in SoundboardPro_Backups folder:")
        try:
            query = f"'{service.backup_folder_id}' in parents and trashed=false"
            results = service.service.files().list(
                q=query,
                pageSize=100,
                fields="files(id, name, size, createdTime)"
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                print("   Folder is empty!")
            else:
                print(f"   Found {len(files)} file(s):")
                for f in files:
                    size = int(f.get('size', 0)) if f.get('size') else 0
                    size_mb = size / 1024 / 1024 if size > 0 else 0
                    print(f"\n   📦 {f['name']}")
                    if size > 0:
                        print(f"      Size: {size_mb:.2f} MB")
                    print(f"      Created: {f.get('createdTime', 'Unknown')}")
        
        except Exception as e:
            print(f"\n❌ Error listing backup folder: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_drive()
