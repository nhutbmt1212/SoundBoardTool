"""Single Instance Lock - Prevent multiple app instances"""
import sys
import os

LOCK_FILE = None
LOCK_HANDLE = None


def acquire_lock() -> bool:
    """Try to acquire single instance lock. Returns True if successful."""
    global LOCK_FILE, LOCK_HANDLE
    
    if sys.platform != 'win32':
        return True
    
    # Use file-based lock (simpler and more reliable)
    try:
        if getattr(sys, 'frozen', False):
            lock_dir = os.path.dirname(sys.executable)
        else:
            lock_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        LOCK_FILE = os.path.join(lock_dir, '.soundboard.lock')
        
        if os.path.exists(LOCK_FILE):
            try:
                with open(LOCK_FILE, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process exists using ctypes
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return False  # Process still running
            except Exception:
                pass  # Lock file is stale
        
        # Write our PID
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        return True
        
    except Exception:
        return True  # If lock fails, allow running


def release_lock():
    """Release the single instance lock"""
    global LOCK_FILE, LOCK_HANDLE
    
    LOCK_HANDLE = None
    
    if LOCK_FILE and os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass
        LOCK_FILE = None


def show_existing_window():
    """Try to bring existing window to front"""
    if sys.platform != 'win32':
        return
    
    try:
        import ctypes
        user32 = ctypes.windll.user32
        
        # Find window by title
        hwnd = user32.FindWindowW(None, "Soundboard Pro")
        if hwnd:
            # Restore and bring to front
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass
