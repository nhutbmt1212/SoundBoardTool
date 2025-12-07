// UI Event Handlers for Soundboard Pro

/**
 * UI event setup and microphone control module
 * Handles UI interactions, drag-drop, and microphone controls
 */
const UIEvents = {
    /**
     * Sets up event listeners for sound cards (click and double-click)
     */
    setupSoundCardEvents() {
        const grid = document.getElementById('sounds-grid');
        const newGrid = grid.cloneNode(true);
        grid.parentNode.replaceChild(newGrid, grid);

        // Single click - select
        newGrid.addEventListener('click', (e) => {
            const card = e.target.closest('.sound-card');
            if (!card) return;

            const name = card.dataset.name;

            if (e.target.closest('.sound-keybind')) {
                e.stopPropagation();
                KeybindEvents.startSoundKeybindRecord(name);
                return;
            }

            SoundEvents.selectSound(name);
        });

        // Double click - play
        newGrid.addEventListener('dblclick', (e) => {
            const card = e.target.closest('.sound-card');
            if (!card) return;
            SoundEvents.playSound(card.dataset.name);
        });
    },

    /**
     * Sets up drag and drop functionality for audio files
     */
    setupDragDrop() {
        const app = document.querySelector('.app');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            app.addEventListener(event, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(event => {
            app.addEventListener(event, () => app.classList.add('drag-over'));
        });

        ['dragleave', 'drop'].forEach(event => {
            app.addEventListener(event, () => app.classList.remove('drag-over'));
        });

        app.addEventListener('drop', async (e) => {
            const files = e.dataTransfer.files;
            if (files.length === 0) return;

            let addedCount = 0;
            for (const file of files) {
                if (Utils.isAudioFile(file)) {
                    const base64 = await Utils.readFileAsBase64(file);
                    const added = await API.addSoundBase64(file.name, base64);
                    if (added) addedCount++;
                }
            }

            if (addedCount === 0) {
                Notifications.warning('No audio files added. Supported: WAV, MP3, OGG, FLAC');
            } else {
                await SoundEvents.refreshSounds();
            }
        });
    },

    // ==================== Microphone Controls ====================

    /**
     * Toggles microphone passthrough on/off
     * @returns {Promise<void>}
     */
    async toggleMic() {
        try {
            AppState.micEnabled = !AppState.micEnabled;
            await API.toggleMicPassthrough(AppState.micEnabled);
            UI.updateMicUI();
        } catch (error) {
            AppState.micEnabled = !AppState.micEnabled; // Revert
        }
    },

    /**
     * Handles microphone volume slider change
     * @param {number} value - New volume value (0-200, maps to 0.0-2.0)
     * @returns {Promise<void>}
     */
    async onMicVolumeChange(value) {
        await API.setMicVolume(parseInt(value) / 100);
    }
};

// Export to global scope
window.UIEvents = UIEvents;
