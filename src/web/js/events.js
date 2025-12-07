// Event Handlers for Soundboard Pro - Main Orchestrator
// This file delegates to specialized event modules

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
    onSoundVolumeChange: (value) => SoundEvents.onVolumeChange(value),
    onSoundNameChange: (value) => SoundEvents.onNameChange(value),
    toggleScreamMode: () => SoundEvents.toggleScreamMode(),
    togglePitchMode: () => SoundEvents.togglePitchMode(),

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
    onYoutubeVolumeChange: (value) => YouTubeEvents.onVolumeChange(value),
    startYoutubeStatusCheck: () => YouTubeEvents.startStatusCheck(),
    startYoutubeKeybindRecording: () => YouTubeEvents.startKeybindRecording(),
    saveYoutubeKeybind: (url, keybind) => YouTubeEvents.saveKeybind(url, keybind),
    toggleYoutubeScreamMode: (url) => YouTubeEvents.toggleScreamMode(url),
    toggleYoutubePitchMode: (url) => YouTubeEvents.togglePitchMode(url),
    onYoutubeNameChange: (value) => YouTubeEvents.onNameChange(value),

    // ==================== Combined Playback Monitoring ====================

    /**
     * Starts periodic playback state checking for both sounds and YouTube
     * This is kept in main file as it monitors both sound and YouTube modules
     */
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

            // Check YouTube playback
            try {
                const ytInfo = await API.getYoutubeInfo();

                if (ytInfo.playing) {
                    UI.updateYoutubeUI(true, ytInfo.title);

                    // Update YouTube grid indicators
                    document.querySelectorAll('.youtube-item').forEach(card => {
                        const cardUrl = card.getAttribute('data-url') || card.dataset.url;
                        const isCardUrl = cardUrl === ytInfo.url;
                        const isPaused = ytInfo.paused;

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
                                const newIcon = isPaused ? IconManager.get('pauseCircle', { size: 32 }) : IconManager.get('playCircle', { size: 32 });
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

        }, 200);
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
window.onYoutubeVolumeChange = (v) => EventHandlers.onYoutubeVolumeChange(v);

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
    }
};
