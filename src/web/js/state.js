// State Management for Soundboard Pro

const AppState = {
    selectedSound: null,
    selectedYoutubeItem: null,
    soundVolumes: {},
    soundKeybinds: {},
    youtubeKeybinds: {},
    soundScreamMode: {},
    soundPitchMode: {},
    youtubeScreamMode: {},
    youtubePitchMode: {},

    soundNames: {},
    youtubeNames: {},
    stopAllKeybind: '',
    isRecordingKeybind: false,
    isRecordingStopKeybind: false,
    micEnabled: false,
    youtubePlayingInterval: null,
    playingCheckInterval: null,
    forceStopped: false,
    currentPlayingSound: null,
    lastStoppedSound: null,
    lastStoppedTime: 0,

    // Getters
    getVolume(name) {
        return this.soundVolumes[name] !== undefined ? this.soundVolumes[name] : 100;
    },

    getKeybind(name) {
        return this.soundKeybinds[name] || '';
    },

    getYoutubeKeybind(url) {
        return this.youtubeKeybinds[url] || '';
    },

    isScreamMode(name) {
        return this.soundScreamMode[name] || false;
    },

    isPitchMode(name) {
        return this.soundPitchMode[name] || false;
    },

    getDisplayName(name) {
        return this.soundNames[name] || name;
    },

    isYoutubeScreamMode(url) {
        return this.youtubeScreamMode[url] || false;
    },

    isYoutubePitchMode(url) {
        return this.youtubePitchMode[url] || false;
    },

    getYoutubeDisplayName(url, defaultTitle) {
        return this.youtubeNames[url] || defaultTitle;
    },

    // Setters
    setVolume(name, value) {
        this.soundVolumes[name] = parseInt(value);
    },

    setKeybind(name, keybind) {
        this.soundKeybinds[name] = keybind;
    },

    setYoutubeKeybind(url, keybind) {
        this.youtubeKeybinds[url] = keybind;
    },

    setScreamMode(name, enabled) {
        this.soundScreamMode[name] = enabled;
    },

    setPitchMode(name, enabled) {
        this.soundPitchMode[name] = enabled;
    },

    setDisplayName(name, displayName) {
        if (displayName && displayName !== name) {
            this.soundNames[name] = displayName;
        } else {
            delete this.soundNames[name];
        }
    },

    setYoutubeScreamMode(url, enabled) {
        this.youtubeScreamMode[url] = enabled;
    },

    setYoutubePitchMode(url, enabled) {
        this.youtubePitchMode[url] = enabled;
    },

    setYoutubeDisplayName(url, displayName, originalTitle) {
        if (displayName && displayName !== originalTitle) {
            this.youtubeNames[url] = displayName;
        } else {
            delete this.youtubeNames[url];
        }
    },

    // Remove sound data
    removeSound(name) {
        delete this.soundVolumes[name];
        delete this.soundKeybinds[name];
        delete this.soundScreamMode[name];
        delete this.soundPitchMode[name];
        delete this.soundNames[name];
    },

    // Load from settings object
    loadFromSettings(settings) {
        this.soundVolumes = settings.volumes || {};
        this.soundKeybinds = settings.keybinds || {};
        this.youtubeKeybinds = settings.youtubeKeybinds || {};
        this.soundScreamMode = settings.screamMode || {};
        this.soundPitchMode = settings.pitchMode || {};
        this.soundNames = settings.names || {};
        this.youtubeNames = settings.youtubeNames || {};
        this.stopAllKeybind = settings.stopAllKeybind || '';
        // Backwards compatibility check - if boolean, reset to empty object
        this.youtubeScreamMode = (typeof settings.youtubeScreamMode === 'object') ? settings.youtubeScreamMode : {};
        this.youtubePitchMode = (typeof settings.youtubePitchMode === 'object') ? settings.youtubePitchMode : {};
    },

    // Export to settings object
    toSettings() {
        return {
            volumes: this.soundVolumes,
            keybinds: this.soundKeybinds,
            youtubeKeybinds: this.youtubeKeybinds,
            screamMode: this.soundScreamMode,
            pitchMode: this.soundPitchMode,
            names: this.soundNames,
            youtubeNames: this.youtubeNames,
            stopAllKeybind: this.stopAllKeybind,
            youtubeScreamMode: this.youtubeScreamMode,
            youtubePitchMode: this.youtubePitchMode
        };
    }
};

window.AppState = AppState;
