// Event Handlers for Soundboard Pro - Main Orchestrator
// This file delegates to specialized event modules

// Constants
const PLAYBACK_CHECK_INTERVAL_MS = 200; // Milliseconds between playback status checks
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
            console.log(`[${type.toUpperCase()}] Info:`, JSON.stringify(info));

            const itemSelector = `.${type}-item`;
            const thumbSelector = type === 'youtube' ? '.youtube-thumbnail' : '.sound-thumbnail';

            if (info.playing) {
                console.log(`[${type.toUpperCase()}] Calling updateUIFn(true, '${info.title}', ${info.paused})`);
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
            console.error(`[${type.toUpperCase()}] Error:`, e);
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

        console.log('🚀 Starting playback check interval...');

        AppState.playingCheckInterval = setInterval(async () => {
            console.log('⏰ Playback check tick...');

            // Skip update completely if force stopped
            if (AppState.forceStopped) {
                console.log('⏭️ Skipped (force stopped)');
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

            console.log('🎬 Checking YouTube playback...');
            // Check YouTube playback
            await EventHandlers._updateStreamPlayback('youtube', API.getYoutubeInfo.bind(API), UI.updateYoutubeUI.bind(UI));

            console.log('🎵 Checking TikTok playback...');
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
