"""Global Hotkey Manager - Works even when app is not focused"""
import threading

KEYBOARD_AVAILABLE = False
keyboard = None


def _init_keyboard():
    """Lazy init keyboard module"""
    global KEYBOARD_AVAILABLE, keyboard
    if keyboard is not None:
        return KEYBOARD_AVAILABLE
    
    try:
        import keyboard as kb
        keyboard = kb
        KEYBOARD_AVAILABLE = True
    except Exception:
        KEYBOARD_AVAILABLE = False
    
    return KEYBOARD_AVAILABLE


class HotkeyManager:
    def __init__(self):
        self._hotkeys = {}
        self._registered = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    def _ensure_init(self):
        if not self._initialized:
            self._initialized = True
            _init_keyboard()
    
    def _convert_keybind(self, keybind: str) -> str:
        """Convert our keybind format to keyboard library format"""
        parts = keybind.split(' + ')
        converted = []
        
        for part in parts:
            p = part.lower()
            if p == 'esc':
                p = 'escape'
            elif p.startswith('num'):
                p = 'num ' + p[3:]
            converted.append(p)
        
        return '+'.join(converted)
    
    def register(self, keybind: str, callback):
        """Register a global hotkey"""
        self._ensure_init()
        
        if not KEYBOARD_AVAILABLE or not keybind:
            return False
        
        with self._lock:
            self._unregister_internal(keybind)
            
            try:
                kb_format = self._convert_keybind(keybind)
                import time
                
                # Debounce mechanism
                def debounced_callback(cb=callback):
                    current_time = time.time()
                    last_time = getattr(cb, '_last_trigger_time', 0)
                    if current_time - last_time < 0.3:  # 300ms debounce
                        return
                    cb._last_trigger_time = current_time
                    cb()
                
                self._hotkeys[keybind] = debounced_callback
                keyboard.add_hotkey(kb_format, debounced_callback, suppress=False)
                self._registered[keybind] = kb_format
                return True
            except Exception:
                return False
    
    def _unregister_internal(self, keybind: str):
        """Internal unregister without lock"""
        if not KEYBOARD_AVAILABLE:
            return
        
        if keybind in self._registered:
            try:
                keyboard.remove_hotkey(self._registered[keybind])
            except Exception:
                pass
            del self._registered[keybind]
        
        if keybind in self._hotkeys:
            del self._hotkeys[keybind]
    
    def unregister(self, keybind: str):
        """Unregister a hotkey"""
        self._ensure_init()
        if not KEYBOARD_AVAILABLE:
            return
        
        with self._lock:
            self._unregister_internal(keybind)
    
    def unregister_all(self):
        """Unregister all hotkeys"""
        self._ensure_init()
        if not KEYBOARD_AVAILABLE:
            return
        
        with self._lock:
            for kb_format in list(self._registered.values()):
                try:
                    keyboard.remove_hotkey(kb_format)
                except Exception:
                    pass
            self._registered.clear()
            self._hotkeys.clear()
    
    def update_all(self, keybinds: dict, play_callback, stop_callback, stop_keybind: str = None, 
                   youtube_keybinds: dict = None, play_youtube_callback = None,
                   tiktok_keybinds: dict = None, play_tiktok_callback = None):
        """Update all hotkeys at once"""
        self._ensure_init()
        if not KEYBOARD_AVAILABLE:
            return
        
        self.unregister_all()
        
        # Register sound keybinds
        for name, keybind in keybinds.items():
            if keybind:
                self.register(keybind, lambda n=name: play_callback(n))
        
        # Register YouTube keybinds
        if youtube_keybinds and play_youtube_callback:
            for url, keybind in youtube_keybinds.items():
                if keybind:
                    self.register(keybind, lambda u=url: play_youtube_callback(u))
        
        # Register TikTok keybinds
        if tiktok_keybinds and play_tiktok_callback:
            for url, keybind in tiktok_keybinds.items():
                if keybind:
                    self.register(keybind, lambda u=url: play_tiktok_callback(u))
        
        if stop_keybind:
            self.register(stop_keybind, stop_callback)


hotkey_manager = HotkeyManager()
