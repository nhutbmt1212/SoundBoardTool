// YouTube Event Handlers for Soundboard Pro

/**
 * YouTube event handlers module
 * Handles all YouTube-related user interactions, playback, and item management
 */
const YouTubeEvents = {
    // ==================== Playback Control ====================

    /**
     * Plays a YouTube video from URL input
     * @returns {Promise<void>}
     */
    async play() {
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
            this.startStatusCheck();

            // Show save button
            const saveBtn = document.getElementById('btn-youtube-save');
            if (saveBtn) saveBtn.style.display = 'flex';
        } else {
            UI.setYoutubeError(result.error || 'Failed to play');
        }

        UI.enableYoutubePlayBtn();
    },

    /**
     * Stops currently playing YouTube video
     * @returns {Promise<void>}
     */
    async stop() {
        await API.stopYoutube();
        UI.updateYoutubeUI(false);
    },

    /**
     * Downloads and saves current YouTube video as a sound file
     * @returns {Promise<void>}
     */
    async saveAsSound() {
        const urlInput = document.getElementById('youtube-url');
        const url = urlInput.value.trim();

        if (!url) {
            Notifications.warning('No YouTube URL');
            return;
        }

        const result = await API.saveYoutubeAsSound(url);

        if (result.success) {
            Notifications.success(`Saved as sound: ${result.name}`);
            await SoundEvents.refreshSounds();
        } else {
            Notifications.error(`Failed to save: ${result.error}`);
        }
    },

    // ==================== Items Management ====================

    /**
     * Refreshes YouTube items grid from backend
     * @returns {Promise<void>}
     */
    async refreshItems() {
        const items = await API.getYoutubeItems();
        const info = await API.getYoutubeInfo();
        UI.renderYoutubeGrid(items, info);
        this.setupCardEvents(items);
    },

    /**
     * Selects a YouTube item and shows its details panel
     * @param {Object} item - YouTube item object
     */
    selectItem(item) {
        AppState.selectedYoutubeItem = item;
        UI.selectYoutubeCard(item.url);
        UI.showYoutubePanel(item);
    },

    /**
     * Shows dialog to add a new YouTube item
     */
    showAddDialog() {
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
                        await this.refreshItems();
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

    /**
     * Plays or resumes a YouTube item by URL
     * @param {string} url - YouTube video URL
     * @returns {Promise<void>}
     */
    async playItem(url) {
        if (this.isPlayProcessing) return;
        this.isPlayProcessing = true;

        try {
            // Check if resuming
            const info = await API.getYoutubeInfo();
            if (info.url === url && info.paused) {
                await API.resumeYoutube();
                this.refreshItems();
                return;
            }

            const result = await API.playYoutube(url);
            if (result.success) {
                this.refreshItems();
            }
        } finally {
            setTimeout(() => {
                this.isPlayProcessing = false;
            }, 300);
        }
    },

    /**
     * Pauses currently playing YouTube item
     * @param {string} url - YouTube video URL
     * @returns {Promise<void>}
     */
    async pauseItem(url) {
        await API.pauseYoutube();
        this.refreshItems();
    },

    /**
     * Deletes a YouTube item after user confirmation
     * @param {string} url - YouTube video URL
     * @returns {Promise<void>}
     */
    async deleteItem(url) {
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
                    AppState.selectedYoutubeItem = null;
                    AppState.removeYoutubeItem(url);
                    await SoundEvents.saveSettings();
                    UI.showEmptyPanel();
                    await this.refreshItems();
                    Notifications.success('YouTube item deleted');
                } else {
                    Notifications.error('Failed to delete item');
                }
            }
        });
    },

    // ==================== Settings & Controls ====================

    /**
     * Handles YouTube volume slider change
     * @param {number} value - New volume value (0-100)
     * @returns {Promise<void>}
     */
    async onVolumeChange(value) {
        await API.setYoutubeVolume(parseInt(value) / 100);
        // Update volume display
        const volumeValue = document.getElementById('youtube-volume-value');
        if (volumeValue) {
            volumeValue.textContent = `${value}%`;
        }
    },

    /**
     * Handles YouTube item custom name change
     * @param {string} value - New display name
     */
    onNameChange(value) {
        if (!AppState.selectedYoutubeItem) return;
        const item = AppState.selectedYoutubeItem;
        AppState.setYoutubeDisplayName(item.url, value.trim(), item.title);
        SoundEvents.saveSettings();
        this.refreshItems().then(() => this.selectItem(item));
    },

    /**
     * Toggles scream mode for a YouTube item
     * @param {string} url - YouTube video URL
     */
    toggleScreamMode(url) {
        if (!url) {
            // Fallback try to get from selected item if no URL passed
            if (AppState.selectedYoutubeItem) url = AppState.selectedYoutubeItem.url;
            else return;
        }

        const checkbox = document.getElementById('yt-scream-checkbox');
        const isScream = checkbox.checked;
        AppState.setYoutubeScreamMode(url, isScream);
        SoundEvents.saveSettings();

        // Update label
        const label = checkbox.parentElement.querySelector('.scream-label');
        if (label) label.textContent = isScream ? 'ON - 5000% BOOST!' : 'OFF';

        // Update wave animation
        const wave = document.querySelector('.panel-preview .preview-wave');
        if (wave) wave.classList.toggle('scream-active', isScream);
    },

    /**
     * Toggles pitch mode for a YouTube item
     * @param {string} url - YouTube video URL
     */
    togglePitchMode(url) {
        if (!url) {
            if (AppState.selectedYoutubeItem) url = AppState.selectedYoutubeItem.url;
            else return;
        }

        const checkbox = document.getElementById('yt-pitch-checkbox');
        const isPitch = checkbox.checked;
        AppState.setYoutubePitchMode(url, isPitch);
        SoundEvents.saveSettings();

        // Update label
        const label = checkbox.parentElement.querySelector('.pitch-label');
        if (label) label.textContent = isPitch ? 'ON - HIGH PITCH!' : 'OFF';
    },

    // ==================== Keybind Management ====================

    /**
     * Starts keybind recording for a YouTube item
     */
    startKeybindRecording() {
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
                this.saveKeybind(item.url, '');
            } else {
                const keybind = Utils.buildKeybindString(e);
                this.saveKeybind(item.url, keybind);
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

    /**
     * Saves YouTube item keybind and updates UI
     * @param {string} url - YouTube video URL
     * @param {string} keybind - Keybind string
     * @returns {Promise<void>}
     */
    async saveKeybind(url, keybind) {
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

    // ==================== UI Event Setup ====================

    /**
     * Sets up event listeners for YouTube item cards
     * @param {Array<Object>} items - Array of YouTube item objects
     */
    setupCardEvents(items) {
        const grid = document.getElementById('youtube-grid');

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
                    this.selectItem(item);
                    this.startKeybindRecording();
                    return;
                }
                this.selectItem(item);
            }
        });

        // Double click - play
        grid.addEventListener('dblclick', (e) => {
            const card = e.target.closest('.youtube-item');
            if (!card) return;
            const url = card.dataset.url;
            this.playItem(url);
        });
    },

    // ==================== Status Monitoring ====================

    /**
     * Starts periodic YouTube status checking
     */
    startStatusCheck() {
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
                YouTubeEvents.refreshItems();
            }
        }, 2000);
    }
};

// Export to global scope
window.YouTubeEvents = YouTubeEvents;
