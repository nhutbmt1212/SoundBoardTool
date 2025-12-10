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

        // Prevent browser shortcuts (Extensions, Defaults) except DevTools
        preventBrowserShortcuts();

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

/**
 * Prevents default browser shortcuts and extension interference
 * Allows only DevTools (F12, Ctrl+Shift+I) and app-specific interactions
 */
function preventBrowserShortcuts() {
    window.addEventListener('keydown', (e) => {
        // ALWAYS Allow F12 and Ctrl+Shift+I (DevTools)
        if (e.key === 'F12' || (e.ctrlKey && e.shiftKey && e.key.toUpperCase() === 'I')) {
            return;
        }

        const isInput = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';
        const isControl = e.ctrlKey || e.altKey || e.metaKey;

        // Block ALL Function keys (F1-F12) except F12 (caught above)
        if (e.key.startsWith('F') && e.key.length > 1) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }

        // Handle Ctrl/Cmd modifiers
        if (isControl) {
            // Allow standard editing shortcuts ONLY in Inputs
            if (isInput) {
                const key = e.key.toLowerCase();
                if (
                    (e.ctrlKey && ['c', 'v', 'x', 'a', 'z', 'y'].includes(key)) // Copy, Paste, Cut, Select All, Undo, Redo
                ) {
                    return; // Let it pass to the input
                }
            }

            // BLOCK EVERYTHING ELSE with modifiers (Ctrl+S, Ctrl+P, Ctrl+Shift+S, Alt+F4 capture?, etc)
            // Note: OS level shortcuts like Alt+F4 or Win+D usually bypass browser JS anyway, but we try.
            // Ctrl+Shift+S (User specific complaint) will be caught here.
            e.preventDefault();
            e.stopPropagation();
            return;
        }

        // If no modifiers, let normal keys pass (typing, app hotkeys)

    }, true); // Use Capture phase to Intercept before bubbling
}
