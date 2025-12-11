"""OAuth callback server for handling Google Drive authentication"""
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import eel

# Global state
_callback_received = threading.Event()
_authorization_response = None
_server_thread = None
_server = None


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback"""
    
    def do_GET(self):
        """Handle GET request from OAuth redirect"""
        global _authorization_response
        
        # Get the full URL
        _authorization_response = f"http://localhost:8080{self.path}"
        
        # Parse query parameters
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        # Send response to browser
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Check if it's a successful callback
        if 'code' in params:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Successful</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    }
                    .container {
                        background: white;
                        padding: 3rem;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        text-align: center;
                        max-width: 400px;
                    }
                    .icon {
                        font-size: 4rem;
                        margin-bottom: 1rem;
                    }
                    h1 {
                        color: #2d3748;
                        margin: 0 0 1rem 0;
                        font-size: 1.8rem;
                    }
                    p {
                        color: #718096;
                        margin: 0;
                        font-size: 1.1rem;
                    }
                    .close-note {
                        margin-top: 1.5rem;
                        padding: 1rem;
                        background: #f7fafc;
                        border-radius: 10px;
                        color: #4a5568;
                        font-size: 0.9rem;
                    }
                    .countdown {
                        font-weight: bold;
                        color: #667eea;
                        font-size: 1.2rem;
                    }
                </style>
                <script>
                    let countdown = 2;
                    
                    function updateCountdown() {
                        const el = document.getElementById('countdown');
                        if (el) {
                            el.textContent = countdown;
                        }
                        
                        if (countdown <= 0) {
                            window.close();
                        } else {
                            countdown--;
                            setTimeout(updateCountdown, 1000);
                        }
                    }
                    
                    // Start countdown
                    window.onload = () => {
                        setTimeout(updateCountdown, 1000);
                    };
                </script>
            </head>
            <body>
                <div class="container">
                    <div class="icon">✅</div>
                    <h1>Authorization Successful!</h1>
                    <p>You're now connected to Google Drive</p>
                    <div class="close-note">
                        Closing in <span class="countdown" id="countdown">2</span> seconds...<br>
                        <small>Return to SoundboardPro to continue</small>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Notify app via Eel
            try:
                eel.on_oauth_success()()
            except:
                pass
        else:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authorization Failed</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    }
                    .container {
                        background: white;
                        padding: 3rem;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        text-align: center;
                        max-width: 400px;
                    }
                    .icon {
                        font-size: 4rem;
                        margin-bottom: 1rem;
                    }
                    h1 {
                        color: #2d3748;
                        margin: 0 0 1rem 0;
                        font-size: 1.8rem;
                    }
                    p {
                        color: #718096;
                        margin: 0;
                        font-size: 1.1rem;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="icon">❌</div>
                    <h1>Authorization Failed</h1>
                    <p>Please try again from the app</p>
                </div>
            </body>
            </html>
            """
        
        self.wfile.write(html.encode())
        
        # Signal that callback was received
        _callback_received.set()
    
    def log_message(self, format, *args):
        """Suppress server logs"""
        pass


def start_callback_server():
    """Start OAuth callback server in background"""
    global _server_thread, _server, _callback_received, _authorization_response
    
    # Reset state
    _callback_received.clear()
    _authorization_response = None
    
    def run_server():
        global _server
        try:
            _server = HTTPServer(('localhost', 8080), OAuthCallbackHandler)
            print("[OAuth] Callback server started on http://localhost:8080")
            
            # Handle requests until callback is received
            while not _callback_received.is_set():
                _server.handle_request()
            
            _server.server_close()
            print("[OAuth] Callback server stopped")
        except Exception as e:
            print(f"[OAuth] Callback server error: {e}")
    
    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()


def stop_callback_server():
    """Stop OAuth callback server"""
    global _server, _callback_received
    
    if _server:
        _callback_received.set()  # Signal to stop
        try:
            _server.shutdown()
        except:
            pass


def get_authorization_response():
    """Get the authorization response URL"""
    return _authorization_response


def wait_for_callback(timeout=120):
    """Wait for OAuth callback with timeout"""
    return _callback_received.wait(timeout)
