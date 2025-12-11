"""Test full AppData backup to Google Drive"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.google_drive_service import GoogleDriveService

def test_backup():
    """Test full backup and restore"""
    print("=" * 60)
    print("Google Drive Full Backup Test")
    print("=" * 60)
    
    service = GoogleDriveService()
    
    # Check login status
    status = service.get_status()
    if not status['is_logged_in']:
        print("\n❌ Not logged in!")
        print("   Please run test_google_oauth.py first to login")
        return
    
    print(f"\n✅ Logged in as: {status['user_email']}")
    
    # Test backup
    print("\n[1] Creating full backup...")
    print("    This will backup:")
    print("    - All settings (JSON files)")
    print("    - All sound files")
    print("    - All cache files")
    print("    - Everything in AppData folder")
    
    result = service.backup_all_settings()
    
    if result.get('success'):
        print("\n✅ Backup successful!")
        print(f"   Filename: {result['filename']}")
        print(f"   Size: {result['size']:,} bytes ({result['size'] / 1024 / 1024:.2f} MB)")
        print(f"   Timestamp: {result['timestamp']}")
    else:
        print(f"\n❌ Backup failed: {result.get('error')}")
        return
    
    # List backups
    print("\n[2] Listing all backups...")
    backups = service.list_backups()
    
    if backups:
        print(f"\n✅ Found {len(backups)} backup(s):")
        for i, backup in enumerate(backups, 1):
            size_mb = int(backup.get('size', 0)) / 1024 / 1024
            print(f"   {i}. {backup['name']}")
            print(f"      Size: {size_mb:.2f} MB")
            print(f"      Created: {backup.get('createdTime', 'Unknown')}")
    else:
        print("\n⚠ No backups found")
    
    print("\n" + "=" * 60)
    print("✅ Full backup test completed!")
    print("\n💡 You can now restore from backup in the app")
    print("   or by running: service.restore_all_settings()")
    print("=" * 60)

if __name__ == "__main__":
    test_backup()
