// TikTok Event Handlers for Soundboard Pro

/**
 * TikTok event handlers module
 * Handles all TikTok-related user interactions, playback, and item management
 */
const TikTokEvents = {
    // ==================== Playback Control ====================

    /**
     * Plays a TikTok video from URL input
     * @returns {Promise<void>}
     */
    async play() {
        const urlInput = document.getElementById('tiktok-url');
        const url = urlInput ? urlInput.value.trim() : '';

        if (!url) {
            Notifications.warning('Please enter a TikTok URL');
            return;
        }

        UI.setTikTokLoading();

        const result = await API.playTikTok(url);

        if (result.success) {
            UI.updateTikTokUI(true, result.title);
            this.startStatusCheck();

            // Show save button
            const saveBtn = document.getElementById('btn-tiktok-save');
            if (saveBtn) saveBtn.style.display = 'flex';
        } else {
            UI.setTikTokError(result.error || 'Failed to play');
        }

        UI.enableTikTokPlayBtn();
    },

    /**
     * Stops currently playing TikTok video
     * @returns {Promise<void>}
     */
    async stop() {
        await API.stopTikTok();
        UI.updateTikTokUI(false);
    },

    /**
     * Downloads and saves current TikTok video as a sound file
     * @returns {Promise<void>}
     */
    async saveAsSound() {
        // Fallback to selected item if no generic input
        const item = AppState.selectedTikTokItem;
        const url = item ? item.url : ''; // Or maybe we need input?

        if (!url) {
            Notifications.warning('No TikTok URL selected');
            return;
        }

        const result = await API.saveTikTokAsSound(url);

        if (result.success) {
            Notifications.success(`Saved as sound: ${result.name}`);
            await SoundEvents.refreshSounds();
        } else {
            Notifications.error(`Failed to save: ${result.error}`);
        }
    },

    // ==================== Items Management ====================

    /**
     * Refreshes TikTok items grid from backend
     * @returns {Promise<void>}
     */
    async refreshItems() {
        const items = await API.getTikTokItems();
        const info = await API.getTikTokInfo();
        UI.renderTikTokGrid(items, info);
        this.setupCardEvents(items);
    },

    /**
     * Selects a TikTok item and shows its details panel
     * @param {Object} item - TikTok item object
     */
    selectItem(item) {
        AppState.selectedTikTokItem = item;
        UI.selectTikTokCard(item.url);
        UI.showTikTokPanel(item);
    },

    /**
     * Shows dialog to add a new TikTok item
     */
    showAddDialog() {
        UI.showModal({
            title: 'Add from TikTok',
            body: `
                <div class="input-group">
                    <input type="text" id="tiktok-url-input" class="modal-input" placeholder="Paste TikTok URL here...">
                    <div style="font-size: 12px; color: var(--text-muted);">Supports individual videos.</div>
                </div>
            `,
            confirmText: 'Add Video',
            onConfirm: async () => {
                const input = document.getElementById('tiktok-url-input');
                const url = input.value.trim();

                if (!url) return;

                // Show loading notification
                Notifications.info('Processing TikTok URL...');
                UI.addLoadingTikTokCard(url);

                try {
                    const result = await API.addTikTokItem(url);

                    if (result.success) {
                        Notifications.success(`Added: ${result.title}`);
                        await this.refreshItems();
                    } else {
                        Notifications.error(`Failed: ${result.error}`);
                        UI.removeLoadingTikTokCard(url);
                    }
                } catch (e) {
                    Notifications.error('An error occurred');
                    UI.removeLoadingTikTokCard(url);
                }
            }
        });
    },

    /**
     * Plays or resumes a TikTok item by URL
     * @param {string} url - TikTok video URL
     * @returns {Promise<void>}
     */
    async playItem(url) {
        if (this.isPlayProcessing) return;
        this.isPlayProcessing = true;

        try {
            const info = await API.getTikTokInfo();

            // Check if resuming
            if (info.url === url && info.paused) {
                await API.resumeTikTok();
                this.refreshItems();
                return;
            }

            // Get settings from state
            let volume = AppState.getTikTokVolume(url) / 100;
            const isScream = AppState.isTikTokScreamMode(url);
            const isPitch = AppState.isTikTokPitchMode(url);
            const trimSettings = AppState.getTikTokTrimSettings(url);

            if (isScream) volume = Math.min(volume * 50.0, 50.0);
            const pitch = isPitch ? 1.5 : 1.0;

            const result = await API.playTikTok(url, volume, pitch, trimSettings?.start || 0, trimSettings?.end || 0);
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
     * Pauses currently playing TikTok item
     * @param {string} url - TikTok video URL
     * @returns {Promise<void>}
     */
    async pauseItem(url) {
        await API.pauseTikTok();
        this.refreshItems();
    },

    /**
     * Deletes a TikTok item after user confirmation
     * @param {string} url - TikTok video URL
     * @returns {Promise<void>}
     */
    async deleteItem(url) {
        // Find title for better message
        const item = AppState.selectedTikTokItem;
        const title = item ? item.title : 'this item';

        UI.showModal({
            title: 'Delete TikTok Item',
            body: `Are you sure you want to delete <b>"${Utils.escapeHtml(title)}"</b>?`,
            confirmText: 'Delete',
            onConfirm: async () => {
                const result = await API.deleteTikTokItem(url);
                if (result.success) {
                    AppState.selectedTikTokItem = null;
                    AppState.removeTikTokItem(url);
                    SoundEvents.saveSettings();
                    UI.showEmptyPanel();
                    await this.refreshItems();
                    Notifications.success('TikTok item deleted');
                } else {
                    Notifications.error('Failed to delete item');
                }
            }
        });
    },

    // ==================== Settings & Controls ====================

    /**
     * Handles TikTok volume live update
     * @param {number} value - New volume value (0-100)
     */
    async onVolumeLive(value) {
        if (!AppState.selectedTikTokItem) return;
        const item = AppState.selectedTikTokItem;

        // Update state but DO NOT save to disk yet
        AppState.setTikTokVolume(item.url, value);

        // Update live volume
        await API.setTikTokVolume(parseInt(value) / 100);

        // Update volume display
        const volumeValue = document.getElementById('tiktok-volume-value');
        if (volumeValue) {
            volumeValue.textContent = `${value}%`;
        }
    },

    /**
     * Handles TikTok volume save (on release)
     * @param {number} value - New volume value
     */
    async onVolumeSave(value) {
        // Persist to disk
        SoundEvents.saveSettings();
    },

    /**
     * Handles TikTok item custom name change
     * @param {string} value - New display name
     */
    onNameChange(value) {
        if (!AppState.selectedTikTokItem) return;
        const item = AppState.selectedTikTokItem;
        AppState.setTikTokDisplayName(item.url, value.trim(), item.title);
        SoundEvents.saveSettings();
        this.refreshItems().then(() => this.selectItem(item));
    },

    /**
     * Toggles scream mode for a TikTok item
     * @param {string} url - TikTok video URL
     */
    toggleScreamMode(url) {
        if (!url) {
            // Fallback try to get from selected item if no URL passed
            if (AppState.selectedTikTokItem) url = AppState.selectedTikTokItem.url;
            else return;
        }

        const checkbox = document.getElementById('tt-scream-checkbox');
        const isScream = checkbox.checked;
        AppState.setTikTokScreamMode(url, isScream);
        SoundEvents.saveSettings();

        // Update label
        const label = checkbox.parentElement.querySelector('.scream-label');
        if (label) label.textContent = isScream ? 'ON - 5000% BOOST!' : 'OFF';

        // Update wave animation
        const wave = document.querySelector('.panel-preview .preview-wave');
        if (wave) wave.classList.toggle('scream-active', isScream);
    },

    /**
     * Toggles pitch mode for a TikTok item
     * @param {string} url - TikTok video URL
     */
    togglePitchMode(url) {
        if (!url) {
            if (AppState.selectedTikTokItem) url = AppState.selectedTikTokItem.url;
            else return;
        }

        const checkbox = document.getElementById('tt-pitch-checkbox');
        const isPitch = checkbox.checked;
        AppState.setTikTokPitchMode(url, isPitch);
        SoundEvents.saveSettings();

        // Update label
        const label = checkbox.parentElement.querySelector('.pitch-label');
        if (label) label.textContent = isPitch ? 'ON - HIGH PITCH!' : 'OFF';
    },

    // ==================== Keybind Management ====================

    /**
     * Starts keybind recording for a TikTok item
     */
    startKeybindRecording() {
        const item = AppState.selectedTikTokItem;
        if (!item) return;

        const input = document.getElementById('tt-keybind-input');
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
                input.value = AppState.getTikTokKeybind(item.url);
            }
        };
        setTimeout(() => document.addEventListener('click', clickHandler), 100);
    },

    /**
     * Saves TikTok item keybind and updates UI
     * @param {string} url - TikTok video URL
     * @param {string} keybind - Keybind string
     * @returns {Promise<void>}
     */
    async saveKeybind(url, keybind) {
        AppState.setTikTokKeybind(url, keybind);
        await API.saveSettings(AppState.toSettings());

        // Update Panel UI
        const input = document.getElementById('tt-keybind-input');
        if (input) input.value = keybind;

        // Update Grid UI (Immediate)
        const cards = document.querySelectorAll('.tiktok-item');
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
     * Sets up event listeners for TikTok item cards
     * @param {Array<Object>} items - Array of TikTok item objects
     */
    setupCardEvents(items) {
        const grid = document.getElementById('tiktok-grid');

        // Single click - select
        grid.addEventListener('click', (e) => {
            const card = e.target.closest('.tiktok-item');
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
            const card = e.target.closest('.tiktok-item');
            if (!card) return;
            const url = card.dataset.url;
            this.playItem(url);
        });
    },

    // ==================== Status Monitoring ====================

    /**
     * Starts periodic TikTok status checking
     */
    startStatusCheck() {
        if (AppState.tiktokPlayingInterval) {
            clearInterval(AppState.tiktokPlayingInterval);
        }

        AppState.tiktokPlayingInterval = setInterval(async () => {
            const info = await API.getTikTokInfo();
            // Update UI with paused state if needed, or just stop if not playing
            if (!info.playing) {
                UI.updateTikTokUI(false);
                clearInterval(AppState.tiktokPlayingInterval);
                AppState.tiktokPlayingInterval = null;
                // Refresh grid to remove playing indicators
                TikTokEvents.refreshItems();
            }
        }, 2000);
    }
};

// Export to global scope
window.TikTokEvents = TikTokEvents;
