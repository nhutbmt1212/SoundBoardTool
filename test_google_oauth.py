"""Test script for Google OAuth flow with automatic callback server"""
import sys
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.google_drive_service import GoogleDriveService

# Global variable to store the authorization response
authorization_response = None
server_ready = threading.Event()


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback"""
    
    def do_GET(self):
        """Handle GET request from OAuth redirect"""
        global authorization_response
        
        # Get the full URL
        authorization_response = f"http://localhost:8080{self.path}"
        
        # Send response to browser
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Check if it's a successful callback
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        if 'code' in params:
            html = """
            <html>
            <head><title>Authorization Successful</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1 style="color: #4CAF50;">✅ Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
            </body>
            </html>
            """
        else:
            html = """
            <html>
            <head><title>Authorization Failed</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1 style="color: #f44336;">❌ Authorization Failed</h1>
                <p>Please check the terminal for error details.</p>
            </body>
            </html>
            """
        
        self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        """Suppress server logs"""
        pass


def run_callback_server():
    """Run the OAuth callback server"""
    global server_ready
    
    server = HTTPServer(('localhost', 8080), OAuthCallbackHandler)
    server_ready.set()
    
    # Handle only one request (the OAuth callback)
    server.handle_request()
    server.server_close()


def test_oauth():
    """Test OAuth flow with automatic callback handling"""
    global authorization_response
    
    print("=" * 60)
    print("Google Drive OAuth Test (Automatic)")
    print("=" * 60)
    
    service = GoogleDriveService()
    
    print("\n[1] Starting OAuth flow...")
    auth_url = service.start_oauth_flow()
    
    if not auth_url:
        print("❌ Failed to start OAuth flow!")
        print("   Check that CLIENT_CONFIG is correctly set in google_drive_service.py")
        return
    
    print("✅ OAuth flow started successfully!")
    
    # Start callback server in background
    print("\n[2] Starting callback server on http://localhost:8080...")
    server_thread = threading.Thread(target=run_callback_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready
    server_ready.wait()
    print("✅ Callback server ready!")
    
    # Open browser automatically
    print("\n[3] Opening browser for authorization...")
    print(f"    URL: {auth_url[:80]}...")
    webbrowser.open(auth_url)
    
    print("\n[4] Waiting for authorization...")
    print("    → Please complete the authorization in your browser")
    print("    → You will be redirected back automatically")
    
    # Wait for callback (max 60 seconds)
    server_thread.join(timeout=60)
    
    if not authorization_response:
        print("\n❌ Timeout! No authorization response received.")
        print("   Make sure you completed the authorization in the browser.")
        return
    
    print("\n✅ Authorization response received!")
    
    # Complete OAuth flow
    print("\n[5] Completing OAuth flow...")
    result = service.complete_oauth_flow(authorization_response)
    
    print("\n" + "=" * 60)
    if result.get('success'):
        print("✅ SUCCESS!")
        print(f"   Logged in as: {result.get('email')}")
        
        print("\n[6] Testing backup functionality...")
        status = service.get_status()
        print(f"   Logged in: {status['is_logged_in']}")
        print(f"   Email: {status['user_email']}")
        print(f"   Auto backup: {status['auto_backup_enabled']}")
        
        print("\n✅ Google Drive integration is working!")
        print("\n💡 You can now use the backup feature in the app!")
    else:
        print("❌ FAILED!")
        print(f"   Error: {result.get('error')}")
        print("\n🔍 Troubleshooting:")
        print("   1. Check that your email is added to Test Users in OAuth consent screen")
        print("   2. Verify redirect URI is http://localhost:8080 in Google Cloud Console")
        print("   3. Make sure Google Drive API is enabled")
        print("   4. Try creating new OAuth credentials")
    
    print("=" * 60)


if __name__ == "__main__":
    test_oauth()
