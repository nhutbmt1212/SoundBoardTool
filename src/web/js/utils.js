// Utility Functions for Soundboard Pro

const Utils = {
    // Escape HTML for safe rendering
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Escape for HTML attributes
    escapeAttr(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    },

    // Read file as base64
    readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    // Check if file is audio
    isAudioFile(file) {
        return file.type.startsWith('audio/') || 
               /\.(wav|mp3|ogg|flac|m4a)$/i.test(file.name);
    },

    // Get key name from event.code
    getKeyFromCode(code) {
        if (code.startsWith('Digit')) return code.replace('Digit', '');
        if (code.startsWith('Numpad')) return 'Num' + code.replace('Numpad', '');
        if (code.startsWith('Key')) return code.replace('Key', '');
        if (code.startsWith('F') && !isNaN(code.slice(1))) return code;
        
        const specialKeys = {
            'Space': 'Space', 'Enter': 'Enter', 'Escape': 'Esc',
            'Backspace': 'Backspace', 'Tab': 'Tab',
            'ArrowUp': 'Up', 'ArrowDown': 'Down', 'ArrowLeft': 'Left', 'ArrowRight': 'Right',
            'Delete': 'Delete', 'Insert': 'Insert', 'Home': 'Home', 'End': 'End',
            'PageUp': 'PageUp', 'PageDown': 'PageDown',
            'Minus': '-', 'Equal': '=', 'BracketLeft': '[', 'BracketRight': ']',
            'Backslash': '\\', 'Semicolon': ';', 'Quote': "'",
            'Comma': ',', 'Period': '.', 'Slash': '/', 'Backquote': '`'
        };
        return specialKeys[code] || code;
    },

    // Check if key is a modifier
    isModifierKey(code) {
        return ['ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight', 
                'AltLeft', 'AltRight', 'MetaLeft', 'MetaRight'].includes(code);
    },

    // Build keybind string from event
    buildKeybindString(e) {
        let keybind = '';
        if (e.ctrlKey) keybind += 'Ctrl + ';
        if (e.shiftKey) keybind += 'Shift + ';
        if (e.altKey) keybind += 'Alt + ';
        keybind += this.getKeyFromCode(e.code);
        return keybind;
    },

    // Match keybind with event
    matchKeybind(event, keybind) {
        if (!keybind) return false;
        
        // Single modifier key
        if (keybind === 'Alt') {
            return event.code.startsWith('Alt') && !event.ctrlKey && !event.shiftKey;
        }
        if (keybind === 'Shift') {
            return event.code.startsWith('Shift') && !event.ctrlKey && !event.altKey;
        }
        if (keybind === 'Ctrl') {
            return event.code.startsWith('Control') && !event.shiftKey && !event.altKey;
        }
        
        // Combo keybind
        const parts = keybind.split(' + ');
        const key = parts[parts.length - 1];
        const needShift = parts.includes('Shift');
        const needCtrl = parts.includes('Ctrl');
        const needAlt = parts.includes('Alt');
        
        if (needShift !== event.shiftKey) return false;
        if (needCtrl !== event.ctrlKey) return false;
        if (needAlt !== event.altKey) return false;
        
        return this.getKeyFromCode(event.code) === key;
    }
};

window.Utils = Utils;
