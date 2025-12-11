// Event Handlers for Soundboard Pro - Main Orchestrator
// This file delegates to specialized event modules

// Constants
const PLAYBACK_CHECK_INTERVAL_MS = 1000; // Milliseconds between playback status checks (optimized from 200ms)
const PLAYING_INDICATOR_ICON_SIZE = 32; // Size of play/pause indicator icons

/**
 * Main event handlers object
 * Delegates to specialized modules: SoundEvents, YouTubeEvents, KeybindEvents, UIEvents
 */
const EventHandlers = {
    // ==================== Sound Handlers (delegated to SoundEvents) ====================
    selectSound: (name) => SoundEvents.selectSound(name),
    playSound: (name) => SoundEvents.playSound(name),
    stopAll: () => SoundEvents.stopAll(),
    addSound: () => SoundEvents.addSound(),
    deleteSound: (name) => SoundEvents.deleteSound(name),
    refreshSounds: () => SoundEvents.refreshSounds(),
    saveSettings: () => SoundEvents.saveSettings(),
    onSoundVolumeLive: (value) => SoundEvents.onVolumeLive(value),
    onSoundVolumeSave: (value) => SoundEvents.onVolumeSave(value),
    onSoundNameChange: (value) => SoundEvents.onNameChange(value),
    toggleScreamMode: () => SoundEvents.toggleScreamMode(),
    togglePitchMode: () => SoundEvents.togglePitchMode(),
    toggleLoop: () => SoundEvents.toggleLoop(),
    onTrimChange: () => SoundEvents.onTrimChange(),

    // ==================== Keybind Handlers (delegated to KeybindEvents) ====================
    startKeybindRecord: (name) => KeybindEvents.startSoundKeybindRecord(name),
    startKeybindRecordPanel: () => KeybindEvents.startSoundKeybindRecordPanel(),
    startStopKeybindRecord: () => KeybindEvents.startStopKeybindRecord(),
    handleKeyDown: (e) => KeybindEvents.handleGlobalKeyDown(e),

    // ==================== UI Setup (delegated to UIEvents) ====================
    setupSoundCardEvents: () => UIEvents.setupSoundCardEvents(),
    setupDragDrop: () => UIEvents.setupDragDrop(),
    toggleMic: () => UIEvents.toggleMic(),
    onMicVolumeChange: (value) => UIEvents.onMicVolumeChange(value),

    // ==================== YouTube Handlers (delegated to YouTubeEvents) ====================
    playYoutube: () => YouTubeEvents.play(),
    stopYoutube: () => YouTubeEvents.stop(),
    saveYoutubeAsSound: () => YouTubeEvents.saveAsSound(),
    refreshYoutubeItems: () => YouTubeEvents.refreshItems(),
    setupYoutubeCardEvents: (items) => YouTubeEvents.setupCardEvents(items),
    selectYoutubeItem: (item) => YouTubeEvents.selectItem(item),
    showAddYoutubeDialog: () => YouTubeEvents.showAddDialog(),
    playYoutubeItem: (url) => YouTubeEvents.playItem(url),
    pauseYoutubeItem: (url) => YouTubeEvents.pauseItem(url),
    deleteYoutubeItem: (url) => YouTubeEvents.deleteItem(url),
    onYoutubeVolumeLive: (value) => YouTubeEvents.onVolumeLive(value),
    onYoutubeVolumeSave: (value) => YouTubeEvents.onVolumeSave(value),
    startYoutubeStatusCheck: () => YouTubeEvents.startStatusCheck(),
    startYoutubeKeybindRecording: () => YouTubeEvents.startKeybindRecording(),
    saveYoutubeKeybind: (url, keybind) => YouTubeEvents.saveKeybind(url, keybind),
    toggleYoutubeScreamMode: (url) => YouTubeEvents.toggleScreamMode(url),
    toggleYoutubePitchMode: (url) => YouTubeEvents.togglePitchMode(url),
    toggleYoutubeLoop: (url) => YouTubeEvents.toggleLoop(url),
    onYoutubeNameChange: (value) => YouTubeEvents.onNameChange(value),

    // ==================== TikTok Handlers (delegated to TikTokEvents) ====================
    playTikTok: () => TikTokEvents.play(),
    stopTikTok: () => TikTokEvents.stop(),
    saveTikTokAsSound: () => TikTokEvents.saveAsSound(),
    refreshTikTokItems: () => TikTokEvents.refreshItems(),
    setupTikTokCardEvents: (items) => TikTokEvents.setupCardEvents(items),
    selectTikTokItem: (item) => TikTokEvents.selectItem(item),
    showAddTikTokDialog: () => TikTokEvents.showAddDialog(),
    playTikTokItem: (url) => TikTokEvents.playItem(url),
    pauseTikTokItem: (url) => TikTokEvents.pauseItem(url),
    deleteTikTokItem: (url) => TikTokEvents.deleteItem(url),
    onTikTokVolumeLive: (value) => TikTokEvents.onVolumeLive(value),
    onTikTokVolumeSave: (value) => TikTokEvents.onVolumeSave(value),
    startTikTokStatusCheck: () => TikTokEvents.startStatusCheck(),
    startTikTokKeybindRecording: () => TikTokEvents.startKeybindRecording(),
    saveTikTokKeybind: (url, keybind) => TikTokEvents.saveKeybind(url, keybind),
    toggleTikTokScreamMode: (url) => TikTokEvents.toggleScreamMode(url),
    toggleTikTokPitchMode: (url) => TikTokEvents.togglePitchMode(url),
    toggleTikTokLoop: (url) => TikTokEvents.toggleLoop(url),
    onTikTokNameChange: (value) => TikTokEvents.onNameChange(value),

    // ==================== Helper Functions ====================

    /**
     * Updates playing indicator for a card
     * @private
     */
    _updateCardPlayingIndicator(card, thumbSelector, isPaused) {
        const thumb = card.querySelector(thumbSelector);
        if (!thumb) return;

        let indicator = thumb.querySelector('.playing-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'playing-indicator';
            thumb.appendChild(indicator);
        }

        const iconName = isPaused ? 'pauseCircle' : 'playCircle';
        const newIcon = IconManager.get(iconName, { size: PLAYING_INDICATOR_ICON_SIZE });
        if (indicator.innerHTML !== newIcon) {
            indicator.innerHTML = newIcon;
        }
    },

    /**
     * Updates card playing state (playing/paused classes and indicator)
     * @private
     */
    _updateCardPlayingState(card, isPlaying, isPaused, thumbSelector) {
        if (isPlaying) {
            if (!card.classList.contains('playing')) card.classList.add('playing');
            if (isPaused) {
                if (!card.classList.contains('paused')) card.classList.add('paused');
            } else {
                card.classList.remove('paused');
            }
            this._updateCardPlayingIndicator(card, thumbSelector, isPaused);
        } else {
            card.classList.remove('playing', 'paused');
            const indicator = card.querySelector('.playing-indicator');
            if (indicator) indicator.remove();
        }
    },

    /**
     * Clears all playing indicators from cards
     * @private
     */
    _clearAllCardIndicators(itemSelector) {
        document.querySelectorAll(itemSelector).forEach(card => {
            card.classList.remove('playing', 'paused');
            const indicator = card.querySelector('.playing-indicator');
            if (indicator) indicator.remove();
        });
    },

    /**
     * Updates stream playback UI (YouTube or TikTok)
     * @private
     */
    async _updateStreamPlayback(type, getInfoFn, updateUIFn) {
        try {
            const info = await getInfoFn();
            const itemSelector = `.${type}-item`;
            const thumbSelector = type === 'youtube' ? '.youtube-thumbnail' : '.sound-thumbnail';

            if (info.playing) {
                updateUIFn(true, info.title, info.paused);

                // Update grid indicators
                document.querySelectorAll(itemSelector).forEach(card => {
                    const cardUrl = card.getAttribute('data-url') || card.dataset.url;
                    const isPlaying = cardUrl === info.url;
                    this._updateCardPlayingState(card, isPlaying, info.paused, thumbSelector);
                });
            } else {
                updateUIFn(false);
                this._clearAllCardIndicators(itemSelector);
            }
        } catch (e) {
            console.error(`[${type.toUpperCase()}] Playback check error:`, e);
        }
    },

    /**
     * Toggles stream playback (pause/resume)
     * @private
     */
    async _toggleStreamPlayback(getInfoFn, pauseFn, resumeFn, refreshFn) {
        const info = await getInfoFn();
        if (info.playing || info.paused) {
            if (info.paused) {
                await resumeFn();
            } else {
                await pauseFn();
            }
            refreshFn();
        }
    },

    // ==================== Combined Playback Monitoring ====================

    /**
     * Starts periodic playback state checking for sounds, YouTube, and TikTok
     */
    startPlayingCheck() {
        if (AppState.playingCheckInterval) {
            clearInterval(AppState.playingCheckInterval);
        }

        AppState.playingCheckInterval = setInterval(async () => {
            // Smart polling: Skip if tab is hidden (user not viewing)
            if (document.hidden) {
                return;
            }

            // Skip update completely if force stopped
            if (AppState.forceStopped) {
                return;
            }

            // Check Sound playback
            const playingSound = await API.getPlayingSound();
            if (!AppState.forceStopped) {
                if (!playingSound) {
                    AppState.currentPlayingSound = null;
                    UI.updatePlayingState(null);
                } else {
                    AppState.currentPlayingSound = playingSound;
                    const isPaused = await API.isSoundPaused();
                    UI.updatePlayingState(playingSound, isPaused);
                }
            }

            // Check YouTube playback
            await EventHandlers._updateStreamPlayback('youtube', API.getYoutubeInfo.bind(API), UI.updateYoutubeUI.bind(UI));

            // Check TikTok playback
            await EventHandlers._updateStreamPlayback('tiktok', API.getTikTokInfo.bind(API), UI.updateTikTokUI.bind(UI));

        }, PLAYBACK_CHECK_INTERVAL_MS);
    },

    // ==================== Now Playing Bar Click ====================
    async onNowPlayingClick() {
        const type = UI.currentPlayingType;
        if (!type) return;

        // Get current playing item and show its detail panel
        if (type === 'sound') {
            const playingSound = AppState.currentPlayingSound;
            if (playingSound) {
                AppState.selectedSound = playingSound;
                UI.selectSoundCard(playingSound);
                UI.showSoundPanel(playingSound);
            }
        } else if (type === 'youtube') {
            const ytInfo = await API.getYoutubeInfo();
            if (ytInfo.url) {
                const items = await API.getYoutubeItems();
                const item = items.find(i => i.url === ytInfo.url);
                if (item) {
                    YouTubeEvents.selectItem(item);
                }
            }
        } else if (type === 'tiktok') {
            const ttInfo = await API.getTikTokInfo();
            if (ttInfo.url) {
                const items = await API.getTikTokItems();
                const item = items.find(i => i.url === ttInfo.url);
                if (item) {
                    TikTokEvents.selectItem(item);
                }
            }
        }
    },

    // ==================== Global Playback Toggle ====================
    async togglePlayback() {
        const type = UI.currentPlayingType;
        if (!type) return;

        if (type === 'sound') {
            const isPaused = await API.isSoundPaused();
            if (isPaused) {
                await API.resumeSound();
            } else {
                await API.pauseSound();
            }
        } else if (type === 'youtube') {
            await this._toggleStreamPlayback(
                API.getYoutubeInfo.bind(API),
                API.pauseYoutube.bind(API),
                API.resumeYoutube.bind(API),
                this.refreshYoutubeItems
            );
        } else if (type === 'tiktok') {
            await this._toggleStreamPlayback(
                API.getTikTokInfo.bind(API),
                API.pauseTikTok.bind(API),
                API.resumeTikTok.bind(API),
                this.refreshTikTokItems
            );
        }
    }
};

// Export to global scope
window.EventHandlers = EventHandlers;

// ==================== Global Function Aliases ====================
// These are used by HTML onclick handlers and must remain for backward compatibility

window.addSound = () => EventHandlers.addSound();
window.refreshSounds = () => EventHandlers.refreshSounds();
window.stopAll = () => EventHandlers.stopAll();
window.startStopKeybindRecord = () => EventHandlers.startStopKeybindRecord();
window.toggleMic = () => EventHandlers.toggleMic();
window.onMicVolumeChange = (v) => EventHandlers.onMicVolumeChange(v);
window.playYoutube = () => EventHandlers.playYoutube();
window.stopYoutube = () => EventHandlers.stopYoutube();
window.saveYoutubeAsSound = () => EventHandlers.saveYoutubeAsSound();
window.onYoutubeVolumeLive = (v) => EventHandlers.onYoutubeVolumeLive(v);
window.onYoutubeVolumeSave = (v) => EventHandlers.onYoutubeVolumeSave(v);
window.showAddYoutubeDialog = () => EventHandlers.showAddYoutubeDialog();
window.refreshYoutubeItems = () => EventHandlers.refreshYoutubeItems();
window.playYoutubeItem = (url) => EventHandlers.playYoutubeItem(url);
window.pauseYoutubeItem = (url) => EventHandlers.pauseYoutubeItem(url);
window.deleteYoutubeItem = (url) => EventHandlers.deleteYoutubeItem(url);
window.playTikTok = () => EventHandlers.playTikTok();
window.stopTikTok = () => EventHandlers.stopTikTok();
window.saveTikTokAsSound = () => EventHandlers.saveTikTokAsSound();
window.onTikTokVolumeLive = (v) => EventHandlers.onTikTokVolumeLive(v);
window.onTikTokVolumeSave = (v) => EventHandlers.onTikTokVolumeSave(v);
window.showAddTikTokDialog = () => EventHandlers.showAddTikTokDialog();
window.refreshTikTokItems = () => EventHandlers.refreshTikTokItems();
window.playTikTokItem = (url) => EventHandlers.playTikTokItem(url);
window.pauseTikTokItem = (url) => EventHandlers.pauseTikTokItem(url);
window.deleteTikTokItem = (url) => EventHandlers.deleteTikTokItem(url);

// ==================== Backup Settings ====================

window.showBackupSettings = async () => {
    await BackupEvents.init();

    const status = BackupEvents.currentStatus;
    const isLoggedIn = status && status.is_logged_in;
    const userName = status && status.user_name;
    const userEmail = status && status.user_email;
    const lastBackup = status && status.last_backup_time
        ? new Date(status.last_backup_time).toLocaleString('vi-VN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        })
        : 'Chưa có';

    UI.showModal({
        title: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" style="vertical-align: middle; margin-right: 8px;">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14z" fill="currentColor"/>
                    <path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zm0 8c-1.65 0-3-1.35-3-3s1.35-3 3-3 3 1.35 3 3-1.35 3-3 3z" fill="currentColor"/>
                </svg>Backup & Settings`,
        body: `
            <div class="backup-settings-premium">
                ${isLoggedIn ? `
                    <!-- User Profile Card -->
                    <div class="backup-user-card">
                        <div class="user-avatar">
                            <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
                                <circle cx="28" cy="28" r="28" fill="url(#avatar-gradient)"/>
                                <path d="M28 28c3.87 0 7-3.13 7-7s-3.13-7-7-7-7 3.13-7 7 3.13 7 7 7zm0 3.5c-5.17 0-9.33 2.09-9.33 4.67V39h18.66v-2.83c0-2.58-4.16-4.67-9.33-4.67z" fill="white" opacity="0.95"/>
                                <defs>
                                    <linearGradient id="avatar-gradient" x1="0" y1="0" x2="56" y2="56">
                                        <stop offset="0%" stop-color="#667eea"/>
                                        <stop offset="100%" stop-color="#764ba2"/>
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                        <div class="user-info">
                            <div class="user-name">${userName || userEmail}</div>
                            ${userName ? `<div class="user-email">${userEmail}</div>` : ''}
                        </div>
                        <div class="google-badge">
                            <svg width="20" height="20" viewBox="0 0 48 48">
                                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                            </svg>
                        </div>
                    </div>

                    <!-- Backup Status -->
                    <div class="backup-status-card">
                        <div class="status-item">
                            <div class="status-icon">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                                    <circle cx="12" cy="12" r="10" stroke="#667eea" stroke-width="2"/>
                                    <path d="M12 6v6l4 2" stroke="#667eea" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                            </div>
                            <div class="status-content">
                                <div class="status-label">Sao lưu lần cuối</div>
                                <div class="status-value">${lastBackup}</div>
                            </div>
                        </div>
                    </div>

                    <!-- Auto Backup Toggle -->
                    <div class="backup-toggle-card">
                        <label class="toggle-label">
                            <input type="checkbox" id="auto-backup-toggle" 
                                ${status.auto_backup_enabled ? 'checked' : ''}
                                onchange="BackupEvents.toggleAutoBackup(this.checked)">
                            <span class="toggle-slider"></span>
                            <div class="toggle-content">
                                <span class="toggle-text">Tự động sao lưu</span>
                                <span class="toggle-desc">Tự động lưu khi có thay đổi cài đặt</span>
                            </div>
                        </label>
                    </div>

                    <!-- Action Buttons -->
                    <div class="backup-actions">
                        <button class="btn-premium btn-backup" onclick="BackupEvents.backup()">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                                <path d="M19 12v7H5v-7M12 3v12m0 0l-4-4m4 4l4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <span>Sao lưu ngay</span>
                        </button>
                        <button class="btn-premium btn-restore" onclick="BackupEvents.restore()">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                                <path d="M5 12v7h14v-7M12 15V3m0 0L8 7m4-4l4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            <span>Khôi phục</span>
                        </button>
                    </div>

                    <!-- Logout Button -->
                    <button class="btn-premium btn-logout" onclick="BackupEvents.logout()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                            <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5m0 0l-5-5m5 5H9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <span>Đăng xuất</span>
                    </button>
                ` : `
                    <!-- Login State -->
                    <div class="backup-login-state">
                        <div class="login-icon">
                            <svg width="80" height="80" viewBox="0 0 80 80" fill="none">
                                <circle cx="40" cy="40" r="40" fill="url(#login-bg-gradient)" opacity="0.15"/>
                                <g transform="translate(16, 16)">
                                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                                    <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                                    <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                                </g>
                                <defs>
                                    <linearGradient id="login-bg-gradient" x1="0" y1="0" x2="80" y2="80">
                                        <stop offset="0%" stop-color="#667eea"/>
                                        <stop offset="100%" stop-color="#764ba2"/>
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>
                        <h3>Kết nối Google Drive</h3>
                        <p>Sao lưu an toàn cài đặt, âm thanh và cấu hình của bạn lên Google Drive. Truy cập mọi lúc, mọi nơi.</p>
                        <button class="btn-premium btn-login" onclick="BackupEvents.showLoginModal()">
                            <svg width="24" height="24" viewBox="0 0 48 48">
                                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                            </svg>
                            <span>Đăng nhập với Google</span>
                        </button>
                    </div>
                `}
            </div>
        `,
        showCancel: false,
        showFooter: false,
        onConfirm: () => {
            UI.closeModal();
        }
    });
};

// ==================== Tab Switching ====================

window.switchTab = (tabName) => {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`content-${tabName}`).classList.add('active');

    // Refresh items when switching to YouTube tab
    if (tabName === 'youtube') {
        EventHandlers.refreshYoutubeItems();
    } else if (tabName === 'tiktok') {
        EventHandlers.refreshTikTokItems();
    }
};
