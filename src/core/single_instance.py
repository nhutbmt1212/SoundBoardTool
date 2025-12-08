"""Single Instance Lock - Prevent multiple app instances"""
import sys
import os
import atexit

LOCK_FILE = None
_browser_pid = None


def acquire_lock() -> bool:
    """Try to acquire single instance lock. Returns True if successful."""
    global LOCK_FILE
    
    if sys.platform != 'win32':
        return True
    
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
                
                # Check if process exists
                if _is_process_running(pid):
                    return False  # Process still running
            except Exception:
                pass  # Lock file is stale
        
        # Write our PID
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        # Register cleanup on exit
        atexit.register(release_lock)
        
        return True
        
    except Exception:
        return True


def _is_process_running(pid: int) -> bool:
    """Check if a process is running"""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
    except Exception:
        pass
    return False


def release_lock():
    """Release the single instance lock"""
    global LOCK_FILE
    
    if LOCK_FILE and os.path.exists(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except Exception:
            pass
        LOCK_FILE = None
    
    # Kill browser if still running
    kill_browser()


def kill_browser():
    """Kill the browser process started by eel"""
    global _browser_pid
    
    # Method 1: Kill by stored PID
    if _browser_pid:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, False, _browser_pid)  # PROCESS_TERMINATE
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
        except Exception:
            pass
        _browser_pid = None
    
    # Method 2: Kill by process tree (all children)
    try:
        import subprocess
        pid = os.getpid()
        subprocess.run(
            f'wmic process where (ParentProcessId={pid}) delete',
            shell=True, capture_output=True, timeout=3
        )
    except Exception:
        pass
    
    # Method 3: Kill browser with --app flag (eel's browser)
    try:
        import subprocess
        subprocess.run(
            ['taskkill', '/F', '/IM', 'msedge.exe', '/FI', 'WINDOWTITLE eq Dalit*'],
            capture_output=True, timeout=2
        )
        subprocess.run(
            ['taskkill', '/F', '/IM', 'chrome.exe', '/FI', 'WINDOWTITLE eq Dalit*'],
            capture_output=True, timeout=2
        )
    except Exception:
        pass


def set_browser_pid(pid: int):
    """Store browser PID for cleanup"""
    global _browser_pid
    _browser_pid = pid


def show_existing_window():
    """Try to bring existing window to front"""
    if sys.platform != 'win32':
        return
    
    try:
        import ctypes
        user32 = ctypes.windll.user32
        
        # Find window by partial title match
        def enum_callback(hwnd, results):
            import ctypes
            length = user32.GetWindowTextLengthW(hwnd) + 1
            buffer = ctypes.create_unicode_buffer(length)
            user32.GetWindowTextW(hwnd, buffer, length)
            if 'Dalit' in buffer.value:
                results.append(hwnd)
            return True
        
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.py_object)
        results = []
        user32.EnumWindows(WNDENUMPROC(enum_callback), results)
        
        if results:
            hwnd = results[0]
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass
