"""Core Utility Functions"""
import os
import shutil

def find_ffmpeg():
    """Find ffmpeg executable in common system paths"""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    
    common_paths = [
        os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\WinGet\Packages'),
        r'C:\ffmpeg\bin',
        r'C:\Program Files\ffmpeg\bin',
        os.path.expandvars(r'%USERPROFILE%\ffmpeg\bin'),
    ]
    
    for base in common_paths:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if 'ffmpeg.exe' in files:
                    return os.path.join(root, 'ffmpeg.exe')
    
    return None

def kill_process_tree(pid: int):
    """Robustly kill a process tree on Windows"""
    if not pid:
        return
        
    import subprocess
    try:
        # Use taskkill to force kill process tree
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
            timeout=2,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
    except Exception:
        pass
