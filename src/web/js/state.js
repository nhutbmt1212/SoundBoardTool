// State Management for Soundboard Pro

const AppState = {
    selectedSound: null,
    soundVolumes: {},
    soundKeybinds: {},
    soundScreamMode: {},
    soundPitchMode: {},
    soundNames: {},
    stopAllKeybind: '',
    isRecordingKeybind: false,
    isRecordingStopKeybind: false,
    micEnabled: false,
    youtubePlayingInterval: null,
    playingCheckInterval: null,
    forceStopped: false,

    // Getters
    getVolume(name) {
        return this.soundVolumes[name] !== undefined ? this.soundVolumes[name] : 100;
    },

    getKeybind(name) {
        return this.soundKeybinds[name] || '';
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

    // Setters
    setVolume(name, value) {
        this.soundVolumes[name] = parseInt(value);
    },

    setKeybind(name, keybind) {
        this.soundKeybinds[name] = keybind;
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
        this.soundScreamMode = settings.screamMode || {};
        this.soundPitchMode = settings.pitchMode || {};
        this.soundNames = settings.names || {};
        this.stopAllKeybind = settings.stopAllKeybind || '';
    },

    // Export to settings object
    toSettings() {
        return {
            volumes: this.soundVolumes,
            keybinds: this.soundKeybinds,
            screamMode: this.soundScreamMode,
            pitchMode: this.soundPitchMode,
            names: this.soundNames,
            stopAllKeybind: this.stopAllKeybind
        };
    }
};

window.AppState = AppState;
