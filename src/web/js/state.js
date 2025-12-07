// State Management for Soundboard Pro

/**
 * Application state manager
 * Centralizes all application state and provides getters/setters for state access
 */
const AppState = {
    // ==================== Core State Properties ====================

    /** @type {string|null} Currently selected sound name */
    selectedSound: null,

    /** @type {Object|null} Currently selected YouTube item */
    selectedYoutubeItem: null,

    /** @type {Object<string, number>} Sound volume levels (0-100) */
    soundVolumes: {},

    /** @type {Object<string, string>} Sound keybind mappings */
    soundKeybinds: {},

    /** @type {Object<string, string>} YouTube keybind mappings */
    youtubeKeybinds: {},

    /** @type {Object<string, boolean>} Sound scream mode states */
    soundScreamMode: {},

    /** @type {Object<string, boolean>} Sound pitch mode states */
    soundPitchMode: {},

    /** @type {Object<string, boolean>} YouTube scream mode states */
    youtubeScreamMode: {},

    /** @type {Object<string, boolean>} YouTube pitch mode states */
    youtubePitchMode: {},

    /** @type {Object<string, string>} Custom sound display names */
    soundNames: {},

    /** @type {Object<string, string>} Custom YouTube display names */
    youtubeNames: {},

    /** @type {Object<string, {start: number, end: number}>} Sound trim settings (start/end in seconds) */
    soundTrimSettings: {},

    /** @type {string} Global stop all keybind */
    stopAllKeybind: '',

    /** @type {boolean} Whether currently recording a keybind */
    isRecordingKeybind: false,

    /** @type {boolean} Whether currently recording stop keybind */
    isRecordingStopKeybind: false,

    /** @type {boolean} Microphone passthrough enabled state */
    micEnabled: false,

    /** @type {number|null} YouTube status check interval ID */
    youtubePlayingInterval: null,

    /** @type {number|null} Playing state check interval ID */
    playingCheckInterval: null,

    /** @type {boolean} Force stop flag to prevent UI flicker */
    forceStopped: false,

    /** @type {string|null} Currently playing sound name */
    currentPlayingSound: null,

    /** @type {string|null} Last stopped sound name */
    lastStoppedSound: null,

    /** @type {number} Timestamp of last sound stop */
    lastStoppedTime: 0,

    // ==================== Sound Getters ====================

    /**
     * Gets volume level for a sound
     * @param {string} name - Sound name
     * @returns {number} Volume level (0-100), defaults to 100
     */
    getVolume(name) {
        return this.soundVolumes[name] !== undefined ? this.soundVolumes[name] : 100;
    },

    /**
     * Gets keybind for a sound
     * @param {string} name - Sound name
     * @returns {string} Keybind string or empty string
     */
    getKeybind(name) {
        return this.soundKeybinds[name] || '';
    },

    /**
     * Checks if scream mode is enabled for a sound
     * @param {string} name - Sound name
     * @returns {boolean} True if scream mode enabled
     */
    isScreamMode(name) {
        return this.soundScreamMode[name] || false;
    },

    /**
     * Checks if pitch mode is enabled for a sound
     * @param {string} name - Sound name
     * @returns {boolean} True if pitch mode enabled
     */
    isPitchMode(name) {
        return this.soundPitchMode[name] || false;
    },

    /**
     * Gets custom display name for a sound
     * @param {string} name - Sound name
     * @returns {string} Custom display name or original name
     */
    getDisplayName(name) {
        return this.soundNames[name] || name;
    },

    /**
     * Gets trim settings for a sound
     * @param {string} name - Sound name
     * @returns {{start: number, end: number}|null} Trim settings or null if not set
     */
    getTrimSettings(name) {
        return this.soundTrimSettings[name] || null;
    },

    // ==================== Sound Setters ====================

    /**
     * Sets volume level for a sound
     * @param {string} name - Sound name
     * @param {number} value - Volume level (0-100)
     */
    setVolume(name, value) {
        this.soundVolumes[name] = parseInt(value);
    },

    /**
     * Sets keybind for a sound
     * @param {string} name - Sound name
     * @param {string} keybind - Keybind string
     */
    setKeybind(name, keybind) {
        this.soundKeybinds[name] = keybind;
    },

    /**
     * Sets scream mode for a sound
     * @param {string} name - Sound name
     * @param {boolean} enabled - True to enable scream mode
     */
    setScreamMode(name, enabled) {
        this.soundScreamMode[name] = enabled;
    },

    /**
     * Sets pitch mode for a sound
     * @param {string} name - Sound name
     * @param {boolean} enabled - True to enable pitch mode
     */
    setPitchMode(name, enabled) {
        this.soundPitchMode[name] = enabled;
    },

    /**
     * Sets custom display name for a sound
     * @param {string} name - Sound name
     * @param {string} displayName - Custom display name (empty to reset)
     */
    setDisplayName(name, displayName) {
        if (displayName && displayName !== name) {
            this.soundNames[name] = displayName;
        } else {
            delete this.soundNames[name];
        }
    },

    /**
     * Sets trim settings for a sound
     * @param {string} name - Sound name
     * @param {number} start - Start time in seconds
     * @param {number} end - End time in seconds (0 = no trim)
     */
    setTrimSettings(name, start, end) {
        if (start > 0 || end > 0) {
            this.soundTrimSettings[name] = { start: start || 0, end: end || 0 };
        } else {
            delete this.soundTrimSettings[name];
        }
    },

    /**
     * Removes all data for a sound
     * @param {string} name - Sound name to remove
     */
    removeSound(name) {
        delete this.soundVolumes[name];
        delete this.soundKeybinds[name];
        delete this.soundScreamMode[name];
        delete this.soundPitchMode[name];
        delete this.soundNames[name];
    },

    // ==================== YouTube Getters ====================

    /**
     * Gets keybind for a YouTube item
     * @param {string} url - YouTube URL
     * @returns {string} Keybind string or empty string
     */
    getYoutubeKeybind(url) {
        return this.youtubeKeybinds[url] || '';
    },

    /**
     * Checks if scream mode is enabled for a YouTube item
     * @param {string} url - YouTube URL
     * @returns {boolean} True if scream mode enabled
     */
    isYoutubeScreamMode(url) {
        return this.youtubeScreamMode[url] || false;
    },

    /**
     * Checks if pitch mode is enabled for a YouTube item
     * @param {string} url - YouTube URL
     * @returns {boolean} True if pitch mode enabled
     */
    isYoutubePitchMode(url) {
        return this.youtubePitchMode[url] || false;
    },

    /**
     * Gets custom display name for a YouTube item
     * @param {string} url - YouTube URL
     * @param {string} defaultTitle - Default title to use if no custom name
     * @returns {string} Custom display name or default title
     */
    getYoutubeDisplayName(url, defaultTitle) {
        return this.youtubeNames[url] || defaultTitle;
    },

    // ==================== YouTube Setters ====================

    /**
     * Sets keybind for a YouTube item
     * @param {string} url - YouTube URL
     * @param {string} keybind - Keybind string
     */
    setYoutubeKeybind(url, keybind) {
        this.youtubeKeybinds[url] = keybind;
    },

    /**
     * Sets scream mode for a YouTube item
     * @param {string} url - YouTube URL
     * @param {boolean} enabled - True to enable scream mode
     */
    setYoutubeScreamMode(url, enabled) {
        this.youtubeScreamMode[url] = enabled;
    },

    /**
     * Sets pitch mode for a YouTube item
     * @param {string} url - YouTube URL
     * @param {boolean} enabled - True to enable pitch mode
     */
    setYoutubePitchMode(url, enabled) {
        this.youtubePitchMode[url] = enabled;
    },

    /**
     * Sets custom display name for a YouTube item
     * @param {string} url - YouTube URL
     * @param {string} displayName - Custom display name (empty to reset)
     * @param {string} originalTitle - Original title for comparison
     */
    setYoutubeDisplayName(url, displayName, originalTitle) {
        if (displayName && displayName !== originalTitle) {
            this.youtubeNames[url] = displayName;
        } else {
            delete this.youtubeNames[url];
        }
    },

    // ==================== Persistence ====================

    /**
     * Loads state from settings object
     * @param {Object} settings - Settings object from backend
     */
    loadFromSettings(settings) {
        this.soundVolumes = settings.volumes || {};
        this.soundKeybinds = settings.keybinds || {};
        this.youtubeKeybinds = settings.youtubeKeybinds || {};
        this.soundScreamMode = settings.screamMode || {};
        this.soundPitchMode = settings.pitchMode || {};
        this.soundNames = settings.names || {};
        this.youtubeNames = settings.youtubeNames || {};
        this.soundTrimSettings = settings.trimSettings || {};
        this.stopAllKeybind = settings.stopAllKeybind || '';

        // Backwards compatibility check - if boolean, reset to empty object
        this.youtubeScreamMode = (typeof settings.youtubeScreamMode === 'object')
            ? settings.youtubeScreamMode
            : {};
        this.youtubePitchMode = (typeof settings.youtubePitchMode === 'object')
            ? settings.youtubePitchMode
            : {};
    },

    /**
     * Exports state to settings object for persistence
     * @returns {Object} Settings object ready for backend storage
     */
    toSettings() {
        return {
            volumes: this.soundVolumes,
            keybinds: this.soundKeybinds,
            youtubeKeybinds: this.youtubeKeybinds,
            screamMode: this.soundScreamMode,
            pitchMode: this.soundPitchMode,
            names: this.soundNames,
            youtubeNames: this.youtubeNames,
            trimSettings: this.soundTrimSettings,
            stopAllKeybind: this.stopAllKeybind,
            youtubeScreamMode: this.youtubeScreamMode,
            youtubePitchMode: this.youtubePitchMode
        };
    }
};

window.AppState = AppState;
