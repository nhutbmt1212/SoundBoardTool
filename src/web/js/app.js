// Soundboard Pro - Main Entry Point

document.addEventListener('DOMContentLoaded', async () => {
    await init();
});

async function init() {
    // Initialize UI
    UI.initIcons();
    
    // Load settings
    const settings = await API.loadSettings();
    AppState.loadFromSettings(settings);
    
    // Render sounds
    await EventHandlers.refreshSounds();
    
    // Update UI
    UI.updateStopKeybindUI();
    
    // Setup event listeners
    document.addEventListener('keydown', (e) => EventHandlers.handleKeyDown(e));
    EventHandlers.setupDragDrop();
    
    // Initialize mic status
    AppState.micEnabled = await API.isMicEnabled();
    UI.updateMicUI();
    
    // Start playing check
    EventHandlers.startPlayingCheck();
}
