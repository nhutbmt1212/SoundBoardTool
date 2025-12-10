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
     * Checks if a keybind is already in use
     * @param {string} keybind - The keybind string to check
     * @param {string} currentType - Type of current item ('sound', 'youtube', 'tiktok', 'stop')
     * @param {string} currentId - ID of current item (name or url)
     * @returns {Object|null} Duplicate info or null
     */
    checkDuplicate(keybind, currentType, currentId) {
        if (!keybind) return null;

        // Check Stop All
        if (AppState.stopAllKeybind === keybind && (currentType !== 'stop' || currentId !== 'stop')) {
            return { type: 'Action', name: 'Stop All' };
        }

        // Check Sounds
        for (const [name, bind] of Object.entries(AppState.soundKeybinds)) {
            if (bind === keybind && (currentType !== 'sound' || currentId !== name)) {
                return { type: 'Sound', name: AppState.getDisplayName(name) };
            }
        }

        // Check YouTube
        for (const [url, bind] of Object.entries(AppState.youtubeKeybinds)) {
            if (bind === keybind && (currentType !== 'youtube' || currentId !== url)) {
                return { type: 'YouTube', name: AppState.getYoutubeDisplayName(url, 'YouTube Item') };
            }
        }

        // Check TikTok
        for (const [url, bind] of Object.entries(AppState.tiktokKeybinds)) {
            if (bind === keybind && (currentType !== 'tiktok' || currentId !== url)) {
                return { type: 'TikTok', name: AppState.getTikTokDisplayName(url, 'TikTok Item') };
            }
        }

        return null;
    },

    /**
     * Saves the stop keybind
     * @param {string} keybind 
     */
    saveStopKeybind(keybind) {
        AppState.stopAllKeybind = keybind;
        SoundEvents.saveSettings();

        const el = document.getElementById('stop-keybind');
        if (el) el.classList.remove('recording');
        UI.updateStopKeybindUI();

        AppState.isRecordingStopKeybind = false;
    },

    /**
     * Saves a sound keybind
     * @param {string} name 
     * @param {string} keybind 
     */
    saveSoundKeybind(name, keybind) {
        AppState.setKeybind(name, keybind);
        SoundEvents.saveSettings();

        const input = document.getElementById('keybind-input');
        if (input) {
            input.value = keybind;
            input.classList.remove('recording');
        }

        SoundEvents.refreshSounds().then(() => SoundEvents.selectSound(name));
        AppState.isRecordingKeybind = false;
    },

    /**
     * Global keyboard event handler
     * Handles keybind recording and keybind triggers
     * @param {KeyboardEvent} e - Keyboard event
     */
    /**
     * Removes keybind from a specific item
     * @param {string} type - 'sound', 'youtube', 'tiktok', 'stop'
     * @param {string} id - Name or URL
     */
    removeKeybind(type, id) {
        if (type === 'stop') {
            this.saveStopKeybind('');
        } else if (type === 'sound') {
            AppState.setKeybind(id, '');
            // Refresh UI if the removed one is not the currently selected one?
            // Actually saveSoundKeybind refreshes sounds.
            // But we might be recording for Sound A, and removing from Sound B.
            // We need to save settings.
            SoundEvents.saveSettings();
            // If the sound panel for Sound B is open (unlikely if we are recording A), we'd need to update it.
            // If Sound B is in the grid, we need to update the grid.
            SoundEvents.refreshSounds();
        } else if (type === 'youtube') {
            AppState.setYoutubeKeybind(id, '');
            SoundEvents.saveSettings();
            YouTubeEvents.refreshItems();
        } else if (type === 'tiktok') {
            AppState.setTikTokKeybind(id, '');
            SoundEvents.saveSettings();
            TikTokEvents.refreshItems();
        }
    },

    /**
     * Removes keybind based on input ID (from UI click)
     * @param {string} inputId 
     */
    removeCurrentKeybind(inputId) {
        if (inputId === 'keybind-input') {
            // Sound
            if (AppState.selectedSound) {
                this.saveSoundKeybind(AppState.selectedSound, '');
            }
        } else if (inputId === 'yt-keybind-input') {
            // YouTube
            if (AppState.selectedYoutubeItem) {
                YouTubeEvents.saveKeybind(AppState.selectedYoutubeItem.url, '');
            }
        } else if (inputId === 'tt-keybind-input') {
            // TikTok
            if (AppState.selectedTikTokItem) {
                TikTokEvents.saveKeybind(AppState.selectedTikTokItem.url, '');
            }
        } else if (inputId === 'stop-keybind') { // Although this is usually a div, checking just in case
            this.saveStopKeybind('');
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

            const keybind = Utils.buildKeybindString(e);

            if (keybind === 'Escape') {
                this.saveStopKeybind('');
                return;
            }

            const duplicate = this.checkDuplicate(keybind, 'stop', 'stop');
            if (duplicate) {
                UI.showModal({
                    title: 'Duplicate Keybind',
                    body: `Keybind <b>${keybind}</b> is already used by <b>${duplicate.type}: ${duplicate.name}</b>.<br>Do you want to transfer it to <b>Stop All</b>?`,
                    confirmText: 'Transfer',
                    onConfirm: () => {
                        // Remove from old owner
                        // We need the ID of the duplicate.
                        // Wait, checkDuplicate returns { type, name }. It doesn't return the ID/URL needed for removal.
                        // We need to fix checkDuplicate to return the ID.
                        // Assuming checkDuplicate is updated to return ID.

                        // For now let's implement the logic assuming we can find it.
                        // Re-find to get ID? Or update checkDuplicate first.
                        // Using a helper to find ID again is safe.
                        this._transferKeybind(keybind, 'stop', 'stop');
                    },
                    onCancel: () => {
                        AppState.isRecordingStopKeybind = false;
                        UI.updateStopKeybindUI();
                        const el = document.getElementById('stop-keybind');
                        if (el) el.classList.remove('recording');
                    }
                });
                return;
            }

            this.saveStopKeybind(keybind);
            return;
        }

        // Recording sound keybind
        if (AppState.isRecordingKeybind && AppState.selectedSound) {
            e.preventDefault();
            if (Utils.isModifierKey(e.code)) return;

            const keybind = Utils.buildKeybindString(e);

            if (keybind === 'Escape') {
                this.saveSoundKeybind(AppState.selectedSound, '');
                return;
            }

            const duplicate = this.checkDuplicate(keybind, 'sound', AppState.selectedSound);
            if (duplicate) {
                UI.showModal({
                    title: 'Duplicate Keybind',
                    body: `Keybind <b>${keybind}</b> is already used by <b>${duplicate.type}: ${duplicate.name}</b>.<br>Do you want to transfer it to <b>${AppState.getDisplayName(AppState.selectedSound)}</b>?`,
                    confirmText: 'Transfer',
                    onConfirm: () => {
                        this._transferKeybind(keybind, 'sound', AppState.selectedSound);
                    },
                    onCancel: () => {
                        AppState.isRecordingKeybind = false;
                        const input = document.getElementById('keybind-input');
                        if (input) {
                            input.classList.remove('recording');
                            input.value = AppState.getKeybind(AppState.selectedSound);
                        }
                    }
                });
                return;
            }

            this.saveSoundKeybind(AppState.selectedSound, keybind);
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
     * Helper to transfer keybind from old owner to new owner
     * @private
     */
    _transferKeybind(keybind, newType, newId) {
        // Find who has it
        // We need to iterate again to find the ID to remove.
        // STOP ALL
        if (AppState.stopAllKeybind === keybind) {
            this.saveStopKeybind('');
        }

        // SOUNDS
        for (const [name, bind] of Object.entries(AppState.soundKeybinds)) {
            if (bind === keybind) {
                AppState.setKeybind(name, '');
            }
        }

        // YOUTUBE
        for (const [url, bind] of Object.entries(AppState.youtubeKeybinds)) {
            if (bind === keybind) {
                AppState.setYoutubeKeybind(url, '');
            }
        }

        // TIKTOK
        for (const [url, bind] of Object.entries(AppState.tiktokKeybinds)) {
            if (bind === keybind) {
                AppState.setTikTokKeybind(url, '');
            }
        }

        // Now save to new owner
        if (newType === 'stop') {
            this.saveStopKeybind(keybind);
        } else if (newType === 'sound') {
            this.saveSoundKeybind(newId, keybind);
        } else if (newType === 'youtube') {
            YouTubeEvents.saveKeybind(newId, keybind);
        } else if (newType === 'tiktok') {
            TikTokEvents.saveKeybind(newId, keybind);
        }

        // Force refresh all grids/UIs since we might have touched anything
        SoundEvents.refreshSounds();
        YouTubeEvents.refreshItems();
        TikTokEvents.refreshItems();
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
