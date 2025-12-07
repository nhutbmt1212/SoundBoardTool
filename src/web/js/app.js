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
        console.log('[App] Initializing Soundboard Pro...');

        // Initialize UI components
        UI.initIcons();
        console.log('[App] UI icons initialized');

        // Load and apply settings
        const settings = await API.loadSettings();
        AppState.loadFromSettings(settings);
        console.log('[App] Settings loaded');

        // Render sounds with retry mechanism
        await loadSoundsWithRetry();
        console.log('[App] Sounds loaded');

        // Update UI with loaded state
        UI.updateStopKeybindUI();

        // Setup global event listeners
        document.addEventListener('keydown', (e) => EventHandlers.handleKeyDown(e));
        EventHandlers.setupDragDrop();
        console.log('[App] Event listeners registered');

        // Initialize microphone status
        AppState.micEnabled = await API.isMicEnabled();
        UI.updateMicUI();
        console.log('[App] Microphone status initialized');

        // Start periodic playback state checks
        EventHandlers.startPlayingCheck();
        console.log('[App] Playback monitoring started');

        console.log('[App] Initialization complete');
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
            console.log(`[App] Sounds loaded successfully on attempt ${attempt + 1}`);
            return;
        }

        // Wait before retry with exponential backoff
        if (attempt < MAX_ATTEMPTS - 1) {
            const delay = 300 * (attempt + 1);
            console.log(`[App] No sounds found, retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }

    console.log('[App] No sounds found after retries (this is normal for first run)');
}
