// UI Rendering for Soundboard Pro

const UI = {
    // Initialize icons
    initIcons() {
        const setIcon = (id, name, size = 20) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = IconManager.get(name, { size });
        };

        setIcon('title-icon', 'music', 28);
        setIcon('stop-icon', 'stop', 20);
        setIcon('placeholder-icon', 'waveform', 64);

        // Tabs
        setIcon('icon-tab-sounds', 'music', 16);
        setIcon('icon-tab-sounds', 'music', 16);
        setIcon('icon-tab-youtube', 'youtube', 16);
        setIcon('icon-tab-tiktok', 'tiktok', 16);

        // Actions
        setIcon('icon-btn-add', 'add', 14);
        setIcon('icon-btn-refresh', 'refresh', 14);
        setIcon('icon-btn-add-yt', 'add', 14);
        setIcon('icon-btn-add-tt', 'add', 14);
        setIcon('icon-btn-refresh-yt', 'refresh', 14);
        setIcon('icon-btn-refresh-tt', 'refresh', 14);

        // Mic
        setIcon('mic-icon', 'micOff', 16); // Default off
    },

    // Update Stop All keybind UI
    updateStopKeybindUI() {
        const el = document.getElementById('stop-keybind');
        const textEl = document.getElementById('stop-keybind-text');
        if (el && textEl) {
            if (AppState.stopAllKeybind) {
                el.classList.add('has-bind');
                textEl.textContent = AppState.stopAllKeybind;
            } else {
                el.classList.remove('has-bind');
                textEl.textContent = 'Add keybind';
            }
        }
    },

    // Render sound grid (delegated to GridRenderer)
    renderSoundGrid(sounds) {
        GridRenderer.renderSoundGrid(sounds);
    },

    // Render single sound card (delegated to CardRenderer)
    renderSoundCard(name) {
        return CardRenderer.renderSoundCard(name);
    },

    // Select sound card visually
    selectSoundCard(name) {
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.toggle('selected', card.dataset.name === name);
        });
    },

    // Show right panel for selected sound (delegated to PanelRenderer)
    showSoundPanel(name) {
        PanelRenderer.renderSoundPanel(name);
    },

    // Show empty panel (delegated to PanelRenderer)
    showEmptyPanel() {
        PanelRenderer.showEmptyPanel();
    },

    // Unified Now Playing Bar
    currentPlayingType: null,

    updateNowPlaying(title, type, isPlaying, isPaused = false) {
        const iconContainer = document.getElementById('now-playing-icon');
        const titleEl = document.getElementById('now-playing-title');
        const statusIcon = document.getElementById('now-playing-status-icon');
        const progressFill = document.getElementById('now-playing-progress');

        // Clear state: no title and not playing
        if (!title && !isPlaying) {
            // Only clear if the current type matches the request type
            // This prevents 'youtube' stop event from clearing 'sound' play event logic if race condition
            if (this.currentPlayingType === type || this.currentPlayingType === null) {
                titleEl.textContent = 'Select a sound...';
                titleEl.classList.remove('active');
                iconContainer.classList.remove('playing');
                iconContainer.innerHTML = IconManager.get('waveform', { size: 24 });
                statusIcon.innerHTML = '';
                progressFill.style.width = '0%';
                this.currentPlayingType = null;
            }
            return;
        }

        // If we have a title (even with isPlaying=false for loading states), show it
        if (title) {
            // Active playing, paused, or loading state
            this.currentPlayingType = type;

            // Update Title
            titleEl.textContent = title || 'Unknown Track';
            titleEl.classList.add('active');

            // Update Icon based on type
            let typeIcon = 'waveform';
            if (type === 'youtube') typeIcon = 'youtube';
            if (type === 'tiktok') typeIcon = 'tiktok';

            iconContainer.innerHTML = IconManager.get(typeIcon, { size: 24 });

            // Update Playing Status
            // Determine control icon
            let controlIcon = 'play';
            if (isPlaying && !isPaused) {
                controlIcon = 'pause';
            }

            const btnHtml = `
                <button class="btn-control-main" id="btn-now-playing-toggle" onclick="EventHandlers.togglePlayback()">
                    ${IconManager.get(controlIcon, { size: 16 })}
                </button>
            `;

            // Only update if changed to prevent flicker/event loss (though onclick is inline)
            if (statusIcon.innerHTML !== btnHtml) {
                statusIcon.innerHTML = btnHtml;
            }

            if (isPlaying && !isPaused) {
                iconContainer.classList.add('playing');
                progressFill.style.width = '100%';
            } else {
                iconContainer.classList.remove('playing');
                progressFill.style.width = '0%';
            }
        }
    },

    // Update playing state on cards (Local Sounds)
    updatePlayingState(playingSound, isPaused = false) {
        document.querySelectorAll('.sound-card').forEach(card => {
            // Only add playing class if playingSound matches, otherwise remove
            if (playingSound && card.dataset.name === playingSound) {
                card.classList.add('playing');
            } else {
                card.classList.remove('playing');
            }
        });

        if (playingSound) {
            const displayName = AppState.getDisplayName(playingSound);
            this.updateNowPlaying(displayName, 'sound', true, isPaused);
        } else {
            this.updateNowPlaying(null, 'sound', false);
        }
    },

    // Clear all playing states
    clearPlayingStates() {
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.remove('playing');
        });
    },

    // Update mic UI
    updateMicUI() {
        const btn = document.getElementById('btn-mic');
        const text = document.getElementById('mic-text');
        if (btn && text) {
            btn.classList.toggle('active', AppState.micEnabled);
            const iconEl = document.getElementById('mic-icon');
            if (iconEl) iconEl.innerHTML = IconManager.get(AppState.micEnabled ? 'mic' : 'micOff', { size: 16 });
            text.textContent = AppState.micEnabled ? 'Mic ON' : 'Mic OFF';
        }
    },

    // Update YouTube UI
    updateYoutubeUI(playing, title = '', paused = false) {
        if (title) {
            this.updateNowPlaying(title, 'youtube', playing, paused);
        } else if (!playing && !title) {
            this.updateNowPlaying(null, 'youtube', false);
        }
    },

    // Set YouTube loading state
    setYoutubeLoading() {
        this.updateNowPlaying('Loading YouTube...', 'youtube', false);
        const playBtn = document.getElementById('btn-youtube-play');
        if (playBtn) playBtn.disabled = true;
    },

    // Set YouTube error state
    setYoutubeError(error) {
        const infoEl = document.getElementById('youtube-info');
        infoEl.innerHTML = IconManager.get('warning', { size: 14 }) + ' ' + Utils.escapeHtml(error);
        infoEl.className = 'youtube-info error';
    },

    // Enable YouTube play button
    enableYoutubePlayBtn() {
        document.getElementById('btn-youtube-play').disabled = false;
    },

    // Select YouTube card visually
    selectYoutubeCard(url) {
        document.querySelectorAll('.youtube-item').forEach(card => {
            card.classList.toggle('selected', card.dataset.url === url);
        });
    },

    // Show panel for YouTube item (delegated to PanelRenderer)
    showYoutubePanel(item) {
        PanelRenderer.renderYoutubePanel(item);
    },

    // Render YouTube grid (delegated to GridRenderer)
    renderYoutubeGrid(items, info) {
        GridRenderer.renderStreamGrid('youtube', items, info);
    },

    // YouTube loading cards (delegated to CardRenderer)
    addLoadingYoutubeCard(url) {
        CardRenderer.addLoadingCard('youtube', url);
    },

    updateYoutubeProgress(url, percent) {
        CardRenderer.updateProgress('youtube', url, percent);
    },

    removeLoadingYoutubeCard(url) {
        CardRenderer.removeLoadingCard('youtube', url);
    },

    renderYoutubeItem(item, info) {
        return CardRenderer.renderStreamItem('youtube', item, info);
    },

    // Update TikTok UI
    updateTikTokUI(playing, title = '', paused = false) {
        if (title) {
            this.updateNowPlaying(title, 'tiktok', playing, paused);
        } else if (!playing && !title) {
            this.updateNowPlaying(null, 'tiktok', false);
        }
    },

    // Set TikTok loading state
    setTikTokLoading() {
        this.updateNowPlaying('Loading TikTok...', 'tiktok', false);
    },

    // Set TikTok error state
    setTikTokError(error) {
        const infoEl = document.getElementById('tiktok-info');
        infoEl.innerHTML = IconManager.get('warning', { size: 14 }) + ' ' + Utils.escapeHtml(error);
        infoEl.className = 'tiktok-info error';
    },



    // Select TikTok card visually
    selectTikTokCard(url) {
        document.querySelectorAll('.tiktok-item').forEach(card => {
            card.classList.toggle('selected', card.dataset.url === url);
        });
    },

    // Show panel for TikTok item (delegated to PanelRenderer)
    showTikTokPanel(item) {
        PanelRenderer.renderTikTokPanel(item);
    },

    // Render TikTok grid (delegated to GridRenderer)
    renderTikTokGrid(items, info) {
        GridRenderer.renderStreamGrid('tiktok', items, info);
    },

    // Render single TikTok item (delegated to CardRenderer)
    renderTikTokItem(item, info) {
        return CardRenderer.renderStreamItem('tiktok', item, info);
    },

    // TikTok loading cards (delegated to CardRenderer)
    addLoadingTikTokCard(url) {
        CardRenderer.addLoadingCard('tiktok', url);
    },



    removeLoadingTikTokCard(url) {
        CardRenderer.removeLoadingCard('tiktok', url);
    },

    // Modal
    closeModalCallback: null,

    showModal({ title, body, onConfirm, confirmText = 'Confirm', showCancel = true }) {
        const overlay = document.getElementById('modal-overlay');
        const titleEl = document.getElementById('modal-title');
        const bodyEl = document.getElementById('modal-body');
        const confirmBtn = document.getElementById('modal-confirm-btn');
        const footer = document.getElementById('modal-footer');

        if (!overlay) return;

        titleEl.textContent = title;
        bodyEl.innerHTML = body;
        confirmBtn.textContent = confirmText;

        // Reset footer
        footer.style.display = 'flex';

        // Hide cancel button if needed? Using CSS might be better but let's leave it for now
        // For simple prompts we want Cancel.

        // Clean previous listeners
        const newConfirmBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);

        newConfirmBtn.onclick = () => {
            if (onConfirm) onConfirm();
            this.closeModal();
        };

        // Handle Enter key in inputs
        const inputs = bodyEl.querySelectorAll('input');
        if (inputs.length > 0) {
            inputs[0].focus();
            inputs[0].addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    newConfirmBtn.click();
                }
            });
            // Focus first input after slight delay for animation
            setTimeout(() => inputs[0].focus(), 50);
        }

        overlay.classList.add('active');
        this.closeModalCallback = null; // Reset callback
    },

    closeModal() {
        const overlay = document.getElementById('modal-overlay');
        if (overlay) overlay.classList.remove('active');
        if (this.closeModalCallback) this.closeModalCallback();
    }
};

window.UI = UI;
