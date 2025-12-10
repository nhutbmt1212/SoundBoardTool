// Utility Functions for Soundboard Pro

/**
 * Mapping of keyboard event codes to display names
 * @const {Object<string, string>}
 */
const SPECIAL_KEYS_MAP = {
    'Space': 'Space',
    'Enter': 'Enter',
    'Escape': 'Esc',
    'Backspace': 'Backspace',
    'Tab': 'Tab',
    'ArrowUp': 'Up',
    'ArrowDown': 'Down',
    'ArrowLeft': 'Left',
    'ArrowRight': 'Right',
    'Delete': 'Delete',
    'Insert': 'Insert',
    'Home': 'Home',
    'End': 'End',
    'PageUp': 'PageUp',
    'PageDown': 'PageDown',
    'Minus': '-',
    'Equal': '=',
    'BracketLeft': '[',
    'BracketRight': ']',
    'Backslash': '\\',
    'Semicolon': ';',
    'Quote': "'",
    'Comma': ',',
    'Period': '.',
    'Slash': '/',
    'Backquote': '`'
};

const Utils = {
    // ==================== HTML Safety ====================

    /**
     * Escapes HTML special characters for safe rendering
     * @param {string} text - Text to escape
     * @returns {string} HTML-safe text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Escapes text for safe use in HTML attributes
     * @param {string} text - Text to escape
     * @returns {string} Attribute-safe text
     */
    escapeAttr(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    },

    // ==================== File Operations ====================

    /**
     * Reads a file and converts it to base64 encoding
     * @param {File} file - File object to read
     * @returns {Promise<string>} Base64 encoded file content (without data URI prefix)
     */
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

    /**
     * Checks if a file is an audio file based on MIME type or extension
     * @param {File} file - File object to check
     * @returns {boolean} True if file is audio
     */
    isAudioFile(file) {
        return file.type.startsWith('audio/') ||
            /\.(wav|mp3|ogg|flac|m4a)$/i.test(file.name);
    },

    // ==================== Keyboard Utilities ====================

    /**
     * Converts keyboard event code to human-readable key name
     * @param {string} code - Keyboard event code (e.g., 'KeyA', 'Digit1')
     * @returns {string} Human-readable key name
     */
    getKeyFromCode(code) {
        // Handle digit keys (Digit0-9)
        if (code.startsWith('Digit')) {
            return code.replace('Digit', '');
        }

        // Handle numpad keys
        if (code.startsWith('Numpad')) {
            return 'Num' + code.replace('Numpad', '');
        }

        // Handle letter keys (KeyA-Z)
        if (code.startsWith('Key')) {
            return code.replace('Key', '');
        }

        // Handle function keys (F1-F12)
        if (code.startsWith('F') && !isNaN(code.slice(1))) {
            return code;
        }

        // Handle special keys from map
        return SPECIAL_KEYS_MAP[code] || code;
    },

    /**
     * Checks if a keyboard code represents a modifier key
     * @param {string} code - Keyboard event code
     * @returns {boolean} True if code is a modifier key
     */
    isModifierKey(code) {
        return ['ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight',
            'AltLeft', 'AltRight', 'MetaLeft', 'MetaRight'].includes(code);
    },

    /**
     * Builds a keybind string from keyboard event
     * @param {KeyboardEvent} event - Keyboard event
     * @returns {string} Keybind string (e.g., 'Ctrl + Shift + A')
     */
    buildKeybindString(event) {
        let keybind = '';
        if (event.ctrlKey) keybind += 'Ctrl + ';
        if (event.shiftKey) keybind += 'Shift + ';
        if (event.altKey) keybind += 'Alt + ';
        keybind += this.getKeyFromCode(event.code);
        return keybind;
    },

    /**
     * Matches a single modifier key press
     * @private
     * @param {KeyboardEvent} event - Keyboard event
     * @param {string} keybind - Keybind string to match
     * @returns {boolean} True if event matches the single modifier keybind
     */
    _matchSingleModifierKey(event, keybind) {
        if (keybind === 'Alt') {
            return event.code.startsWith('Alt') && !event.ctrlKey && !event.shiftKey;
        }
        if (keybind === 'Shift') {
            return event.code.startsWith('Shift') && !event.ctrlKey && !event.altKey;
        }
        if (keybind === 'Ctrl') {
            return event.code.startsWith('Control') && !event.shiftKey && !event.altKey;
        }
        return false;
    },

    /**
     * Matches a combo keybind (modifier + key)
     * @private
     * @param {KeyboardEvent} event - Keyboard event
     * @param {string} keybind - Keybind string to match
     * @returns {boolean} True if event matches the combo keybind
     */
    _matchComboKeybind(event, keybind) {
        const parts = keybind.split(' + ');
        const key = parts[parts.length - 1];
        const needShift = parts.includes('Shift');
        const needCtrl = parts.includes('Ctrl');
        const needAlt = parts.includes('Alt');

        if (needShift !== event.shiftKey) return false;
        if (needCtrl !== event.ctrlKey) return false;
        if (needAlt !== event.altKey) return false;

        return this.getKeyFromCode(event.code) === key;
    },

    /**
     * Checks if a keyboard event matches a keybind string
     * @param {KeyboardEvent} event - Keyboard event to check
     * @param {string} keybind - Keybind string to match against
     * @returns {boolean} True if event matches keybind
     */
    matchKeybind(event, keybind) {
        if (!keybind) return false;

        // Check for single modifier key
        if (['Alt', 'Shift', 'Ctrl'].includes(keybind)) {
            return this._matchSingleModifierKey(event, keybind);
        }

        // Check for combo keybind
        return this._matchComboKeybind(event, keybind);
    }
};

window.Utils = Utils;
