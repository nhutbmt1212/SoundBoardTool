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
 * Browser shortcuts that should be blocked
 * Only block specific shortcuts, not all modifier combinations
 * @const {Set<string>}
 */
const BLOCKED_BROWSER_SHORTCUTS = new Set([
    'Ctrl+S',       // Save page
    'Ctrl+Shift+S', // Save as
    'Ctrl+P',       // Print
    'Ctrl+R',       // Reload
    'Ctrl+N',       // New window
    'Ctrl+W',       // Close tab
    'Ctrl+T',       // New tab
    'Ctrl+H',       // History
    'Ctrl+J',       // Downloads
    'Ctrl+D',       // Bookmark
    'Ctrl+U',       // View source
    'Ctrl+G',       // Find next
    'Ctrl+F',       // Find
    'Ctrl+O',       // Open file
    'Alt+Home',     // Home page
    'Alt+Left',     // Back
    'Alt+Right',    // Forward
]);

/**
 * Prevents default browser shortcuts and extension interference
 * Allows keybind recording, DevTools, and app-specific interactions
 */
function preventBrowserShortcuts() {
    window.addEventListener('keydown', (e) => {
        // ALWAYS Allow F12 and Ctrl+Shift+I (DevTools)
        if (e.key === 'F12' || (e.ctrlKey && e.shiftKey && e.key.toUpperCase() === 'I')) {
            return;
        }

        // ALWAYS Allow when recording keybinds - let keybind handler process it
        // Check the flags on AppState AND look for any element with 'recording' class
        const isRecordingAnyKeybind = AppState.isRecordingKeybind ||
            AppState.isRecordingStopKeybind ||
            document.querySelector('.recording');
        if (isRecordingAnyKeybind) {
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
                if (e.ctrlKey && ['c', 'v', 'x', 'a', 'z', 'y'].includes(key)) {
                    return; // Let it pass to the input
                }
            }

            // Build shortcut string to check against blocked list
            let shortcut = '';
            if (e.ctrlKey) shortcut += 'Ctrl+';
            if (e.shiftKey) shortcut += 'Shift+';
            if (e.altKey) shortcut += 'Alt+';
            shortcut += e.key.length === 1 ? e.key.toUpperCase() : e.key;

            // Only block specific browser shortcuts, allow others for app keybinds
            if (BLOCKED_BROWSER_SHORTCUTS.has(shortcut)) {
                e.preventDefault();
                e.stopPropagation();
                return;
            }
        }

        // If no modifiers or not a blocked shortcut, let it pass for app keybind processing

    }, true); // Use Capture phase to Intercept before bubbling
}
