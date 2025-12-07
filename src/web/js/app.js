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

    // Render sounds with retry (backend may not be ready immediately)
    await loadSoundsWithRetry();

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

async function loadSoundsWithRetry() {
    // Try to load sounds, retry if empty
    for (let attempt = 0; attempt < 3; attempt++) {
        await EventHandlers.refreshSounds();

        // Check if sounds were loaded
        const grid = document.getElementById('sounds-grid');
        const hasContent = grid && grid.children.length > 0 && !grid.querySelector('.empty-state');

        if (hasContent) {
            console.log('✓ Sounds loaded successfully');
            return;
        }

        // Wait before retry (exponential backoff)
        if (attempt < 2) {
            const delay = 300 * (attempt + 1);
            console.log(`Retrying sound load in ${delay}ms... (attempt ${attempt + 1}/3)`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }

    console.log('Sound loading completed (may be empty)');
}
