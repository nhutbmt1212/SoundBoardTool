# Google Drive Backup Setup Guide

## 📋 Prerequisites
- Google Account
- Google Cloud Project (free tier is sufficient)

## 🔧 Setup Steps

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project"
3. Name your project (e.g., "SoundboardPro Backup")
4. Click "Create"

### 2. Enable Google Drive API

1. In your project, go to "APIs & Services" > "Library"
2. Search for "Google Drive API"
3. Click on it and press "Enable"

### 3. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure OAuth consent screen:
   - User Type: External
   - App name: SoundboardPro
   - User support email: Your email
   - Developer contact: Your email
   - Save and Continue through all steps
4. Back to Create OAuth client ID:
   - Application type: Desktop app
   - Name: SoundboardPro Desktop
   - Click "Create"
5. Download the JSON file

### 4. Update Application Code

1. Open the downloaded JSON file
2. Copy the `client_id` and `client_secret`
3. Open `src/services/google_drive_service.py`
4. Replace the `CLIENT_CONFIG` dictionary:

```python
CLIENT_CONFIG = {
    "installed": {
        "client_id": "YOUR_CLIENT_ID_HERE.apps.googleusercontent.com",
        "project_id": "your-project-id",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "YOUR_CLIENT_SECRET_HERE",
        "redirect_uris": ["http://localhost:8080"]
    }
}
```

### 5. Install Dependencies

```bash
pip install -r requirements_backup.txt
```

### 6. Test the Integration

1. Run the application
2. Click the Settings button (⚙️) in the bottom right
3. Click "Login with Google"
4. Follow the OAuth flow
5. After successful login, try "Backup Now"

## 🔒 Security Notes

- **Never commit** your `client_secret` to version control
- Consider using environment variables for production:
  ```python
  import os
  CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
  CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
  ```
- The credentials are stored locally in `%APPDATA%/SoundboardPro/backup_config.json`
- Users can logout anytime to clear stored credentials

## 📝 What Gets Backed Up

- `sound_settings.json` - All sound settings, keybinds, volumes, effects
- `soundboard_config.json` - Application configuration

## 🔄 Auto Backup

When enabled, the app will automatically backup to Google Drive whenever:
- Settings are changed
- Keybinds are modified
- Sound effects are updated
- Volume levels are adjusted

## 🐛 Troubleshooting

### "Failed to start OAuth flow"
- Check that Google Drive API is enabled
- Verify CLIENT_CONFIG is correctly set

### "Login failed"
- Make sure redirect URI is `http://localhost:8080`
- Check OAuth consent screen is configured

### "Backup failed"
- Verify you're logged in
- Check internet connection
- Ensure Google Drive has sufficient space

## 📚 Additional Resources

- [Google Drive API Documentation](https://developers.google.com/drive/api/v3/about-sdk)
- [OAuth 2.0 for Desktop Apps](https://developers.google.com/identity/protocols/oauth2/native-app)
