// Soundboard Pro - Main Entry Point

/**
 * Application initialization on DOM ready
 */
document.addEventListener('DOMContentLoaded', async () => {
    Notifications.init();
    await init();
});

/**
 * Initializes the application
 * Sets up UI, loads settings, renders sounds, and starts event listeners
 * @returns {Promise<void>}
 */
async function init() {
    try {

        // Initialize UI components
        UI.initIcons();

        // Load and apply settings
        const settings = await API.loadSettings();
        AppState.loadFromSettings(settings);

        // Render sounds with retry mechanism
        await loadSoundsWithRetry();

        // Update UI with loaded state
        UI.updateStopKeybindUI();

        // Setup global event listeners
        document.addEventListener('keydown', (e) => EventHandlers.handleKeyDown(e));
        EventHandlers.setupDragDrop();

        // Initialize microphone status
        AppState.micEnabled = await API.isMicEnabled();
        UI.updateMicUI();

        // Start periodic playback state checks
        EventHandlers.startPlayingCheck();

    } catch (error) {
        console.error('[App] Initialization failed:', error);
        Notifications.error('Failed to initialize application. Please refresh the page.');
    }
}

/**
 * Loads sounds with retry mechanism
 * Attempts to load sounds up to 3 times with exponential backoff
 * Handles cases where backend may not be ready immediately
 * @returns {Promise<void>}
 */
async function loadSoundsWithRetry() {
    const MAX_ATTEMPTS = 3;

    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        await EventHandlers.refreshSounds();

        // Check if sounds were successfully loaded
        const grid = document.getElementById('sounds-grid');
        const hasContent = grid && grid.children.length > 0 && !grid.querySelector('.empty-state');

        if (hasContent) {
            return;
        }

        // Wait before retry with exponential backoff
        if (attempt < MAX_ATTEMPTS - 1) {
            const delay = 300 * (attempt + 1);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }

}
