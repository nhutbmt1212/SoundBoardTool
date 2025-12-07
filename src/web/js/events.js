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
        if (label) label.textContent = isScream ? 'ON - 5000% BOOST!' : 'OFF';

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
                Notifications.warning('No audio files added. Supported: WAV, MP3, OGG, FLAC');
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
            Notifications.warning('Please enter a YouTube URL');
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
            Notifications.warning('No YouTube URL');
            return;
        }

        const result = await API.saveYoutubeAsSound(url);

        if (result.success) {
            Notifications.success(`Saved as sound: ${result.name}`);
            await this.refreshSounds();
        } else {
            Notifications.error(`Failed to save: ${result.error}`);
        }
    },

    // YouTube Items Management
    async refreshYoutubeItems() {
        const items = await API.getYoutubeItems();
        const info = await API.getYoutubeInfo();
        UI.renderYoutubeGrid(items, info);
        this.setupYoutubeCardEvents(items);
    },

    // Setup YouTube card events
    setupYoutubeCardEvents(items) {
        const grid = document.getElementById('youtube-grid');

        // Single click - select
        // Single click - select
        grid.addEventListener('click', (e) => {
            const card = e.target.closest('.youtube-item');
            if (!card) return;

            const url = card.dataset.url;
            // Find item object
            const item = items.find(i => i.url === url);
            if (item) {
                if (e.target.closest('.sound-keybind')) {
                    e.stopPropagation();
                    this.selectYoutubeItem(item);
                    this.startYoutubeKeybindRecording();
                    return;
                }
                this.selectYoutubeItem(item);
            }
        });

        // Double click - play
        grid.addEventListener('dblclick', (e) => {
            const card = e.target.closest('.youtube-item');
            if (!card) return;
            const url = card.dataset.url;
            this.playYoutubeItem(url);
        });
    },

    selectYoutubeItem(item) {
        AppState.selectedYoutubeItem = item;
        UI.selectYoutubeCard(item.url);
        UI.showYoutubePanel(item);
    },

    showAddYoutubeDialog() {
        UI.showModal({
            title: 'Add from YouTube',
            body: `
                <div class="input-group">
                    <input type="text" id="youtube-url-input" class="modal-input" placeholder="Paste YouTube URL here...">
                    <div style="font-size: 12px; color: var(--text-muted);">Supports individual videos. Playlist support coming soon.</div>
                </div>
            `,
            confirmText: 'Add Video',
            onConfirm: async () => {
                const input = document.getElementById('youtube-url-input');
                const url = input.value.trim();

                if (!url) return;

                // Show loading notification
                Notifications.info('Processing YouTube URL...');
                UI.addLoadingYoutubeCard(url);

                try {
                    const result = await API.addYoutubeItem(url);

                    if (result.success) {
                        Notifications.success(`Added: ${result.title}`);
                        await this.refreshYoutubeItems();
                    } else {
                        Notifications.error(`Failed: ${result.error}`);
                        UI.removeLoadingYoutubeCard(url);
                    }
                } catch (e) {
                    Notifications.error('An error occurred');
                    UI.removeLoadingYoutubeCard(url);
                }
            }
        });
    },

    async playYoutubeItem(url) {
        // Check if resuming
        const info = await API.getYoutubeInfo();
        if (info.url === url && info.paused) {
            await API.resumeYoutube();
            this.refreshYoutubeItems();
            return;
        }

        const result = await API.playYoutube(url);
        if (result.success) {
            this.refreshYoutubeItems();
        }
    },

    async pauseYoutubeItem(url) {
        await API.pauseYoutube();
        this.refreshYoutubeItems();
    },

    async deleteYoutubeItem(url) {
        // Find title for better message
        const item = AppState.selectedYoutubeItem;
        const title = item ? item.title : 'this item';

        UI.showModal({
            title: 'Delete YouTube Item',
            body: `Are you sure you want to delete <b>"${Utils.escapeHtml(title)}"</b>?`,
            confirmText: 'Delete',
            onConfirm: async () => {
                const result = await API.deleteYoutubeItem(url);
                if (result.success) {
                    await this.refreshYoutubeItems();
                    Notifications.success('YouTube item deleted');
                } else {
                    Notifications.error('Failed to delete item');
                }
            }
        });
    },

    async bindYoutubeKey(url) {
        Notifications.info('Press a key to bind...');
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
            // Update UI with paused state if needed, or just stop if not playing
            if (!info.playing) {
                UI.updateYoutubeUI(false);
                clearInterval(AppState.youtubePlayingInterval);
                AppState.youtubePlayingInterval = null;
                // Refresh grid to remove playing indicators
                EventHandlers.refreshYoutubeItems();
            } else {
                // Update grid periodically to show play/pause state
                // This might be too heavy? Maybe just when state changes?
                // For now, let's just accept it might not auto-update pause state if changed externally
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

            // Check Sound playback
            const playingSound = await API.getPlayingSound();
            if (!AppState.forceStopped) {
                if (!playingSound) {
                    AppState.currentPlayingSound = null;
                }
                UI.updatePlayingState(playingSound);
            }

            // Check YouTube playback - do this every tick or every few ticks?
            // Doing it every 100ms might be okay, let's try.
            try {
                const ytInfo = await API.getYoutubeInfo();

                // Only update if state changed significantly to avoid redraws?
                // For now, simple update
                if (ytInfo.playing) {
                    UI.updateYoutubeUI(true, ytInfo.title);
                    // Also update grid if needed
                    // We need to optimize this to not re-render grid constantly
                    // But we can update classes on existing elements
                    document.querySelectorAll('.youtube-item').forEach(card => {
                        const cardUrl = card.getAttribute('data-url') || card.dataset.url;
                        const isCardUrl = cardUrl === ytInfo.url;
                        const isPaused = ytInfo.paused;

                        // Handle active state
                        if (isCardUrl) {
                            if (!card.classList.contains('playing')) card.classList.add('playing');
                            if (isPaused) {
                                if (!card.classList.contains('paused')) card.classList.add('paused');
                            } else {
                                card.classList.remove('paused');
                            }

                            // Update indicator
                            const thumb = card.querySelector('.youtube-thumbnail');
                            if (thumb) {
                                let indicator = thumb.querySelector('.playing-indicator');
                                if (!indicator) {
                                    indicator = document.createElement('div');
                                    indicator.className = 'playing-indicator';
                                    thumb.appendChild(indicator);
                                }
                                // Only update content if needed
                                const newIcon = isPaused ? IconManager.get('pauseCircle', { size: 32 }) : IconManager.get('playCircle', { size: 32 });
                                // Simple check: compare HTML length or something, or just clear and set
                                // Since we are setting innerHTML, let's just do it
                                if (indicator.innerHTML !== newIcon) {
                                    indicator.innerHTML = newIcon;
                                }
                            }
                        } else {
                            if (card.classList.contains('playing')) card.classList.remove('playing');
                            if (card.classList.contains('paused')) card.classList.remove('paused');
                            const indicator = card.querySelector('.playing-indicator');
                            if (indicator) indicator.remove();
                        }
                    });
                } else {
                    UI.updateYoutubeUI(false);
                    // Clear grid indicators
                    document.querySelectorAll('.youtube-item').forEach(card => {
                        card.classList.remove('playing', 'paused');
                        const indicator = card.querySelector('.playing-indicator');
                        if (indicator) indicator.remove();
                    });
                }
            } catch (e) {
                // Ignore errors
            }

        }, 200); // Increased interval to 200ms to reduce load with double check
    },

    // --- YouTube Keybind ---

    startYoutubeKeybindRecording() {
        const item = AppState.selectedYoutubeItem;
        if (!item) return;

        const input = document.getElementById('yt-keybind-input');
        if (!input) return;

        input.value = 'Press any key...';
        input.classList.add('recording');

        const handler = (e) => {
            e.preventDefault();
            e.stopPropagation();

            // Ignore pure modifier keys
            if (Utils.isModifierKey(e.code)) return;

            let key = e.key;
            if (key === 'Escape') {
                this.saveYoutubeKeybind(item.url, '');
            } else {
                const keybind = Utils.buildKeybindString(e);
                this.saveYoutubeKeybind(item.url, keybind);
            }

            document.removeEventListener('keydown', handler);
            input.classList.remove('recording');
        };

        document.addEventListener('keydown', handler);

        // Remove handler if clicking outside
        const clickHandler = (e) => {
            if (e.target !== input) {
                document.removeEventListener('keydown', handler);
                document.removeEventListener('click', clickHandler);
                input.classList.remove('recording');
                input.value = AppState.getYoutubeKeybind(item.url);
            }
        };
        setTimeout(() => document.addEventListener('click', clickHandler), 100);
    },

    async saveYoutubeKeybind(url, keybind) {
        AppState.setYoutubeKeybind(url, keybind);
        await API.saveSettings(AppState.toSettings());

        // Update Panel UI
        const input = document.getElementById('yt-keybind-input');
        if (input) input.value = keybind;

        // Update Grid UI (Immediate)
        const cards = document.querySelectorAll('.youtube-item');
        for (const card of cards) {
            if (card.dataset.url === url) {
                const kbEl = card.querySelector('.sound-keybind');
                if (kbEl) {
                    kbEl.textContent = keybind || 'Add keybind';
                    kbEl.classList.toggle('has-bind', !!keybind);
                }
            }
        }
    },

    // Toggle YouTube scream mode
    toggleYoutubeScreamMode(url) {
        if (!url) {
            // Fallback try to get from selected item if no URL passed (should not happen with new UI)
            if (AppState.selectedYoutubeItem) url = AppState.selectedYoutubeItem.url;
            else return;
        }

        const checkbox = document.getElementById('yt-scream-checkbox');
        const isScream = checkbox.checked;
        AppState.setYoutubeScreamMode(url, isScream);
        this.saveSettings();

        // Update label
        const label = checkbox.parentElement.querySelector('.scream-label');
        if (label) label.textContent = isScream ? 'ON - 5000% BOOST!' : 'OFF';

        // Update wave animation
        const wave = document.querySelector('.panel-preview .preview-wave');
        if (wave) wave.classList.toggle('scream-active', isScream);

        // Notify backend of changed settings? 
        // Currently backend reads settings only when play is triggered.
        // That is fine.
    },

    // Toggle YouTube pitch mode
    toggleYoutubePitchMode(url) {
        if (!url) {
            if (AppState.selectedYoutubeItem) url = AppState.selectedYoutubeItem.url;
            else return;
        }

        const checkbox = document.getElementById('yt-pitch-checkbox');
        const isPitch = checkbox.checked;
        AppState.setYoutubePitchMode(url, isPitch);
        this.saveSettings();

        // Update label
        const label = checkbox.parentElement.querySelector('.pitch-label');
        if (label) label.textContent = isPitch ? 'ON - HIGH PITCH!' : 'OFF';
    },

    // YouTube name change
    onYoutubeNameChange(value) {
        if (!AppState.selectedYoutubeItem) return;
        const item = AppState.selectedYoutubeItem;
        AppState.setYoutubeDisplayName(item.url, value.trim(), item.title);
        this.saveSettings();
        this.refreshYoutubeItems().then(() => this.selectYoutubeItem(item));
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
window.pauseYoutubeItem = (url) => EventHandlers.pauseYoutubeItem(url);
window.deleteYoutubeItem = (url) => EventHandlers.deleteYoutubeItem(url);
window.bindYoutubeKey = (url) => EventHandlers.bindYoutubeKey(url);
window.startYoutubeKeybindRecording = () => EventHandlers.startYoutubeKeybindRecording();
window.saveYoutubeKeybind = (url, keybind) => EventHandlers.saveYoutubeKeybind(url, keybind);
