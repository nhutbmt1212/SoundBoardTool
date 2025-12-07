// Event Handlers for Soundboard Pro

const EventHandlers = {
    // Select sound
    selectSound(name) {
        AppState.selectedSound = name;
        UI.selectSoundCard(name);
        UI.showSoundPanel(name);
    },

    // Play sound
    async playSound(name) {
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

        if (isScream) volume = Math.min(volume * 50.0, 50.0);
        const pitch = isPitch ? 1.5 : 1.0;

        await API.playSound(name, volume, pitch);
    },

    // Stop all sounds
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

    // Add sound
    async addSound() {
        const result = await API.addSoundDialog();
        if (result) await this.refreshSounds();
    },

    // Delete sound
    async deleteSound(name) {
        if (!confirm(`Delete "${name}"?`)) return;

        await API.deleteSound(name);
        AppState.removeSound(name);
        await this.saveSettings();

        AppState.selectedSound = null;
        UI.showEmptyPanel();
        await this.refreshSounds();
    },

    // Refresh sounds
    async refreshSounds() {
        const sounds = await API.getSounds();
        UI.renderSoundGrid(sounds);
        this.setupSoundCardEvents();
    },

    // Save settings
    async saveSettings() {
        await API.saveSettings(AppState.toSettings());
    },

    // Sound volume change
    onSoundVolumeChange(value) {
        if (!AppState.selectedSound) return;
        document.getElementById('volume-value').textContent = value;
        AppState.setVolume(AppState.selectedSound, value);
        this.saveSettings();
    },

    // Sound name change
    onSoundNameChange(value) {
        if (!AppState.selectedSound) return;
        AppState.setDisplayName(AppState.selectedSound, value.trim());
        this.saveSettings();
        this.refreshSounds().then(() => this.selectSound(AppState.selectedSound));
    },

    // Toggle scream mode
    toggleScreamMode() {
        if (!AppState.selectedSound) return;
        const checkbox = document.getElementById('scream-checkbox');
        const isScream = checkbox.checked;
        AppState.setScreamMode(AppState.selectedSound, isScream);
        this.saveSettings();

        // Update label
        const label = document.querySelector('.scream-label');
        if (label) label.textContent = isScream ? 'ON - 5000% BOOST! 💀' : 'OFF';

        // Update wave animation
        const wave = document.querySelector('.preview-wave');
        if (wave) wave.classList.toggle('scream-active', isScream);

        // Update card
        this.refreshSounds().then(() => this.selectSound(AppState.selectedSound));
    },

    // Toggle pitch mode
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

    // Start keybind record for sound
    startKeybindRecord(name) {
        AppState.selectedSound = name;
        this.selectSound(name);
        this.startKeybindRecordPanel();
    },

    // Start keybind record in panel
    startKeybindRecordPanel() {
        if (!AppState.selectedSound) return;
        const input = document.getElementById('keybind-input');
        if (!input) return;

        AppState.isRecordingKeybind = true;
        AppState.isRecordingStopKeybind = false;
        input.classList.add('recording');
        input.value = 'Press a key...';
        input.focus();
    },

    // Start stop keybind record
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

    // Handle keyboard events
    handleKeyDown(e) {
        // Recording Stop All keybind
        if (AppState.isRecordingStopKeybind) {
            e.preventDefault();
            if (Utils.isModifierKey(e.code)) return;

            AppState.stopAllKeybind = Utils.buildKeybindString(e);
            this.saveSettings();

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
            this.saveSettings();

            const input = document.getElementById('keybind-input');
            if (input) {
                input.value = keybind;
                input.classList.remove('recording');
            }

            this.refreshSounds().then(() => this.selectSound(AppState.selectedSound));
            AppState.isRecordingKeybind = false;
            return;
        }

        // Check Stop All keybind
        if (AppState.stopAllKeybind && Utils.matchKeybind(e, AppState.stopAllKeybind)) {
            e.preventDefault();
            this.stopAll();
            return;
        }

        // Check sound keybinds
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
                    this.stopAll();
                } else {
                    this.playSound(name);
                }
                return;
            }
        }
    },

    // Setup sound card events
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
                this.startKeybindRecord(name);
                return;
            }

            this.selectSound(name);
        });

        // Double click - play
        newGrid.addEventListener('dblclick', (e) => {
            const card = e.target.closest('.sound-card');
            if (!card) return;
            this.playSound(card.dataset.name);
        });
    },

    // Setup drag & drop
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
                alert('No audio files added. Supported: WAV, MP3, OGG, FLAC');
            } else {
                await this.refreshSounds();
            }
        });
    },

    // Toggle mic
    async toggleMic() {
        try {
            AppState.micEnabled = !AppState.micEnabled;
            await API.toggleMicPassthrough(AppState.micEnabled);
            UI.updateMicUI();
        } catch (error) {
            AppState.micEnabled = !AppState.micEnabled; // Revert
        }
    },

    // Mic volume change
    async onMicVolumeChange(value) {
        await API.setMicVolume(parseInt(value) / 100);
    },

    // Play YouTube
    async playYoutube() {
        const urlInput = document.getElementById('youtube-url');
        const url = urlInput.value.trim();

        if (!url) {
            alert('Please enter a YouTube URL');
            return;
        }

        UI.setYoutubeLoading();

        const result = await API.playYoutube(url);

        if (result.success) {
            UI.updateYoutubeUI(true, result.title);
            this.startYoutubeStatusCheck();

            // Show save button
            const saveBtn = document.getElementById('btn-youtube-save');
            if (saveBtn) saveBtn.style.display = 'flex';
        } else {
            UI.setYoutubeError(result.error || 'Failed to play');
        }

        UI.enableYoutubePlayBtn();
    },

    // Save YouTube as Sound
    async saveYoutubeAsSound() {
        const urlInput = document.getElementById('youtube-url');
        const url = urlInput.value.trim();

        if (!url) {
            alert('No YouTube URL');
            return;
        }

        const result = await API.saveYoutubeAsSound(url);

        if (result.success) {
            alert(`✓ Saved as sound: ${result.name}`);
            await this.refreshSounds();
        } else {
            alert(`Failed to save: ${result.error}`);
        }
    },

    // YouTube Items Management
    async refreshYoutubeItems() {
        const items = await API.getYoutubeItems();
        UI.renderYoutubeGrid(items);
    },

    async showAddYoutubeDialog() {
        const url = prompt('Enter YouTube URL:');
        if (!url) return;

        const result = await API.addYoutubeItem(url);

        if (result.success) {
            alert(`✓ Added: ${result.title}`);
            await this.refreshYoutubeItems();
        } else {
            alert(`Failed: ${result.error}`);
        }
    },

    async playYoutubeItem(url) {
        const result = await API.playYoutube(url);
        if (result.success) {
            this.refreshYoutubeItems();
        }
    },

    async deleteYoutubeItem(url) {
        if (!confirm('Delete this YouTube item?')) return;

        const result = await API.deleteYoutubeItem(url);
        if (result.success) {
            await this.refreshYoutubeItems();
        }
    },

    async bindYoutubeKey(url) {
        alert('Press a key to bind...');
        // TODO: Implement keybind for YouTube items
    },

    // Stop YouTube
    async stopYoutube() {
        await API.stopYoutube();
        UI.updateYoutubeUI(false);
    },

    // YouTube volume change
    async onYoutubeVolumeChange(value) {
        await API.setYoutubeVolume(parseInt(value) / 100);
        // Update volume display
        const volumeValue = document.getElementById('youtube-volume-value');
        if (volumeValue) {
            volumeValue.textContent = `${value}%`;
        }
    },

    // Start YouTube status check
    startYoutubeStatusCheck() {
        if (AppState.youtubePlayingInterval) {
            clearInterval(AppState.youtubePlayingInterval);
        }

        AppState.youtubePlayingInterval = setInterval(async () => {
            const info = await API.getYoutubeInfo();
            if (!info.playing) {
                UI.updateYoutubeUI(false);
                clearInterval(AppState.youtubePlayingInterval);
                AppState.youtubePlayingInterval = null;
            }
        }, 2000);
    },

    // Start playing check interval
    startPlayingCheck() {
        if (AppState.playingCheckInterval) {
            clearInterval(AppState.playingCheckInterval);
        }

        AppState.playingCheckInterval = setInterval(async () => {
            // Skip update completely if force stopped
            if (AppState.forceStopped) {
                return;
            }
            const playingSound = await API.getPlayingSound();
            // Double check flag after async call
            if (AppState.forceStopped) {
                return;
            }

            // Update current playing sound state
            if (!playingSound) {
                AppState.currentPlayingSound = null;
            }

            UI.updatePlayingState(playingSound);
        }, 100);
    }
};

window.EventHandlers = EventHandlers;

// Global function aliases for HTML onclick handlers
window.addSound = () => EventHandlers.addSound();
window.refreshSounds = () => EventHandlers.refreshSounds();
window.stopAll = () => EventHandlers.stopAll();
window.startStopKeybindRecord = () => EventHandlers.startStopKeybindRecord();
window.toggleMic = () => EventHandlers.toggleMic();
window.onMicVolumeChange = (v) => EventHandlers.onMicVolumeChange(v);
window.playYoutube = () => EventHandlers.playYoutube();
window.stopYoutube = () => EventHandlers.stopYoutube();
window.saveYoutubeAsSound = () => EventHandlers.saveYoutubeAsSound();
window.onYoutubeVolumeChange = (v) => EventHandlers.onYoutubeVolumeChange(v);

// Tab switching
window.switchTab = (tabName) => {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`content-${tabName}`).classList.add('active');

    // Load content if needed
    if (tabName === 'youtube') {
        EventHandlers.refreshYoutubeItems();
    }
};

// YouTube items management
window.showAddYoutubeDialog = () => EventHandlers.showAddYoutubeDialog();
window.refreshYoutubeItems = () => EventHandlers.refreshYoutubeItems();
window.playYoutubeItem = (url) => EventHandlers.playYoutubeItem(url);
window.deleteYoutubeItem = (url) => EventHandlers.deleteYoutubeItem(url);
window.bindYoutubeKey = (url) => EventHandlers.bindYoutubeKey(url);
