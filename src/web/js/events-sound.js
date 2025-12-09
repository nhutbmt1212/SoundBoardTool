// Sound Event Handlers for Soundboard Pro

/**
 * Sound event handlers module
 * Handles all sound-related user interactions and state management
 */
const SoundEvents = {
    /**
     * Selects a sound and displays its details panel
     * @param {string} name - Name of the sound to select
     */
    selectSound(name) {
        AppState.selectedSound = name;
        UI.selectSoundCard(name);
        UI.showSoundPanel(name);
    },

    /**
     * Plays a sound with configured volume and effects
     * @param {string} name - Name of the sound to play
     * @returns {Promise<void>}
     */
    async playSound(name) {
        // Prevent rapid double-clicks (debounce)
        if (this.isPlayProcessing) return;
        this.isPlayProcessing = true;

        try {
            // Track current playing sound
            AppState.currentPlayingSound = name;

            // Immediately add playing class
            document.querySelectorAll('.sound-card').forEach(card => {
                if (card.dataset.name === name) {
                    card.classList.add('playing');
                }
            });

            let volume = AppState.getVolume(name) / 100;
            const isScream = AppState.isScreamMode(name);
            const isPitch = AppState.isPitchMode(name);
            const trimSettings = AppState.getTrimSettings(name);

            if (isScream) volume = Math.min(volume * 50.0, 50.0);
            const pitch = isPitch ? 1.5 : 1.0;

            await API.playSound(name, volume, pitch, trimSettings?.start || 0, trimSettings?.end || 0);
        } finally {
            // Release lock after a short delay
            setTimeout(() => {
                this.isPlayProcessing = false;
            }, 300);
        }
    },

    /**
     * Stops all currently playing sounds
     * @returns {Promise<void>}
     */
    async stopAll() {
        // Set flag to force stop - keep it longer
        AppState.forceStopped = true;
        AppState.currentPlayingSound = null;

        await API.stopAll();
        UI.clearPlayingStates();

        // Reset flag after backend has time to fully stop
        setTimeout(() => {
            AppState.forceStopped = false;
        }, 1000);
    },

    /**
     * Opens file dialog to add a new sound
     * @returns {Promise<void>}
     */
    async addSound() {
        const result = await API.addSoundDialog();
        if (result) await this.refreshSounds();
    },

    /**
     * Deletes a sound after user confirmation
     * @param {string} name - Name of the sound to delete
     */
    deleteSound(name) {
        UI.showModal({
            title: 'Delete Sound',
            body: `Are you sure you want to delete <b>"${Utils.escapeHtml(name)}"</b>?`,
            confirmText: 'Delete',
            onConfirm: async () => {
                await API.deleteSound(name);
                AppState.removeSound(name);
                await this.saveSettings();

                AppState.selectedSound = null;
                UI.showEmptyPanel();
                await this.refreshSounds();
                Notifications.success('Sound deleted');
            }
        });
    },

    /**
     * Refreshes the sound grid from backend
     * @returns {Promise<void>}
     */
    async refreshSounds() {
        const sounds = await API.getSounds();
        UI.renderSoundGrid(sounds);
        UIEvents.setupSoundCardEvents();
    },

    /**
     * Saves current application settings to backend
     * @returns {Promise<void>}
     */
    async saveSettings() {
        await API.saveSettings(AppState.toSettings());
    },

    /**
     * Handles sound volume live update
     * @param {number} value - New volume value (0-100)
     */
    async onVolumeLive(value) {
        if (!AppState.selectedSound) return;
        document.getElementById('volume-value').textContent = value;

        // Update state
        AppState.setVolume(AppState.selectedSound, value);

        // Calculate effective volume (including scream mode)
        let vol = parseInt(value) / 100;
        const isScream = AppState.isScreamMode(AppState.selectedSound);
        if (isScream) vol = Math.min(vol * 50.0, 50.0);

        // Send to backend
        await API.setSoundVolume(vol);
    },

    /**
     * Handles sound volume save (on release)
     * @param {number} value - New volume value
     */
    async onVolumeSave(value) {
        this.saveSettings();
    },

    /**
     * Handles sound custom name change
     * @param {string} value - New display name
     */
    onNameChange(value) {
        if (!AppState.selectedSound) return;
        AppState.setDisplayName(AppState.selectedSound, value.trim());
        this.saveSettings();
        this.refreshSounds().then(() => this.selectSound(AppState.selectedSound));
    },

    /**
     * Toggles scream mode for selected sound
     */
    toggleScreamMode() {
        if (!AppState.selectedSound) return;
        const checkbox = document.getElementById('scream-checkbox');
        const isScream = checkbox.checked;
        AppState.setScreamMode(AppState.selectedSound, isScream);
        this.saveSettings();

        // Update label
        const label = document.querySelector('.scream-label');
        if (label) label.textContent = isScream ? 'ON - 5000% BOOST!' : 'OFF';

        // Update wave animation
        const wave = document.querySelector('.preview-wave');
        if (wave) wave.classList.toggle('scream-active', isScream);

        // Update card
        this.refreshSounds().then(() => this.selectSound(AppState.selectedSound));
    },

    /**
     * Toggles pitch mode for selected sound
     */
    togglePitchMode() {
        if (!AppState.selectedSound) return;
        const checkbox = document.getElementById('pitch-checkbox');
        const isPitch = checkbox.checked;
        AppState.setPitchMode(AppState.selectedSound, isPitch);
        this.saveSettings();

        // Update label
        const label = document.querySelector('.pitch-label');
        if (label) label.textContent = isPitch ? 'ON - HIGH PITCH!' : 'OFF';

        // Update card
        this.refreshSounds().then(() => this.selectSound(AppState.selectedSound));
    },

    /**
     * Handles trim settings change
     */
    onTrimChange() {
        if (!AppState.selectedSound) return;
        const startInput = document.getElementById('trim-start');
        const endInput = document.getElementById('trim-end');

        const start = parseFloat(startInput.value) || 0;
        const end = parseFloat(endInput.value) || 0;

        AppState.setTrimSettings(AppState.selectedSound, start, end);
        this.saveSettings();
    }
};

// Export to global scope
window.SoundEvents = SoundEvents;
