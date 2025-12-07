// Keybind Event Handlers for Soundboard Pro

/**
 * Keybind recording and matching module
 * Handles all keybind-related functionality for sounds and global actions
 */
const KeybindEvents = {
    /**
     * Starts keybind recording for a sound
     * @param {string} name - Name of the sound
     */
    startSoundKeybindRecord(name) {
        AppState.selectedSound = name;
        SoundEvents.selectSound(name);
        this.startSoundKeybindRecordPanel();
    },

    /**
     * Starts keybind recording in the panel UI
     */
    startSoundKeybindRecordPanel() {
        if (!AppState.selectedSound) return;
        const input = document.getElementById('keybind-input');
        if (!input) return;

        AppState.isRecordingKeybind = true;
        AppState.isRecordingStopKeybind = false;
        input.classList.add('recording');
        input.value = 'Press a key...';
        input.focus();
    },

    /**
     * Starts recording for the global stop-all keybind
     */
    startStopKeybindRecord() {
        AppState.isRecordingStopKeybind = true;
        AppState.isRecordingKeybind = false;

        const el = document.getElementById('stop-keybind');
        const textEl = document.getElementById('stop-keybind-text');
        if (el && textEl) {
            el.classList.add('recording');
            textEl.textContent = 'Press a key...';
        }
    },

    /**
     * Global keyboard event handler
     * Handles keybind recording and keybind triggers
     * @param {KeyboardEvent} e - Keyboard event
     */
    handleGlobalKeyDown(e) {
        // Recording Stop All keybind
        if (AppState.isRecordingStopKeybind) {
            e.preventDefault();
            if (Utils.isModifierKey(e.code)) return;

            AppState.stopAllKeybind = Utils.buildKeybindString(e);
            SoundEvents.saveSettings();

            const el = document.getElementById('stop-keybind');
            if (el) el.classList.remove('recording');
            UI.updateStopKeybindUI();

            AppState.isRecordingStopKeybind = false;
            return;
        }

        // Recording sound keybind
        if (AppState.isRecordingKeybind && AppState.selectedSound) {
            e.preventDefault();
            if (Utils.isModifierKey(e.code)) return;

            const keybind = Utils.buildKeybindString(e);
            AppState.setKeybind(AppState.selectedSound, keybind);
            SoundEvents.saveSettings();

            const input = document.getElementById('keybind-input');
            if (input) {
                input.value = keybind;
                input.classList.remove('recording');
            }

            SoundEvents.refreshSounds().then(() => SoundEvents.selectSound(AppState.selectedSound));
            AppState.isRecordingKeybind = false;
            return;
        }

        // Check Stop All keybind
        if (AppState.stopAllKeybind && Utils.matchKeybind(e, AppState.stopAllKeybind)) {
            e.preventDefault();
            SoundEvents.stopAll();
            return;
        }

        // Check sound keybinds
        this._matchSoundKeybinds(e);
    },

    /**
     * Matches and triggers sound keybinds
     * @private
     * @param {KeyboardEvent} e - Keyboard event
     */
    _matchSoundKeybinds(e) {
        for (const [name, bind] of Object.entries(AppState.soundKeybinds)) {
            if (Utils.matchKeybind(e, bind)) {
                e.preventDefault();

                // Prevent rapid toggle - if we just stopped this sound, don't play it again
                const now = Date.now();
                if (AppState.lastStoppedSound === name && (now - AppState.lastStoppedTime) < 500) {
                    return;
                }

                // Toggle: if this sound is playing, stop it; otherwise play it
                if (AppState.currentPlayingSound === name) {
                    AppState.lastStoppedSound = name;
                    AppState.lastStoppedTime = now;
                    SoundEvents.stopAll();
                } else {
                    SoundEvents.playSound(name);
                }
                return;
            }
        }
    }
};

// Export to global scope
window.KeybindEvents = KeybindEvents;
