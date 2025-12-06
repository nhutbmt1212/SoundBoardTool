// API Layer - Eel Communication for Soundboard Pro

const API = {
    // Settings
    async loadSettings() {
        try {
            return await eel.get_settings()();
        } catch (e) {
            console.log('No saved settings');
            return {};
        }
    },

    async saveSettings(settings) {
        try {
            await eel.save_settings(settings)();
        } catch (e) {
            console.error('Error saving settings:', e);
        }
    },

    // Sounds
    async getSounds() {
        try {
            return await eel.get_sounds()();
        } catch (e) {
            console.error('Error getting sounds:', e);
            return [];
        }
    },

    async playSound(name, volume, pitch) {
        try {
            await eel.play_sound(name, volume, pitch)();
        } catch (e) {
            console.error('Error playing sound:', e);
        }
    },

    async stopAll() {
        try {
            await eel.stop_all()();
        } catch (e) {
            console.error('Error stopping sounds:', e);
        }
    },

    async getPlayingSound() {
        try {
            return await eel.get_playing_sound()();
        } catch (e) {
            return null;
        }
    },

    async addSoundDialog() {
        try {
            return await eel.add_sound_dialog()();
        } catch (e) {
            console.error('Error adding sound:', e);
            return false;
        }
    },

    async addSoundBase64(filename, base64) {
        try {
            return await eel.add_sound_base64(filename, base64)();
        } catch (e) {
            console.error('Error adding sound:', e);
            return false;
        }
    },

    async deleteSound(name) {
        try {
            await eel.delete_sound(name)();
        } catch (e) {
            console.error('Error deleting sound:', e);
        }
    },

    // Microphone
    async isMicEnabled() {
        try {
            return await eel.is_mic_enabled()();
        } catch (e) {
            console.log('Mic status check failed');
            return false;
        }
    },

    async toggleMicPassthrough(enabled) {
        try {
            await eel.toggle_mic_passthrough(enabled)();
        } catch (e) {
            console.error('Error toggling mic:', e);
            throw e;
        }
    },

    async setMicVolume(volume) {
        try {
            await eel.set_mic_volume(volume)();
        } catch (e) {
            console.error('Error setting mic volume:', e);
        }
    },

    // YouTube
    async playYoutube(url) {
        try {
            return await eel.play_youtube(url)();
        } catch (e) {
            console.error('YouTube error:', e);
            return { success: false, error: e.toString() };
        }
    },

    async stopYoutube() {
        try {
            await eel.stop_youtube()();
        } catch (e) {
            console.error('Error stopping YouTube:', e);
        }
    },

    async getYoutubeInfo() {
        try {
            return await eel.get_youtube_info()();
        } catch (e) {
            return { playing: false };
        }
    },

    async setYoutubeVolume(volume) {
        try {
            await eel.set_youtube_volume(volume)();
        } catch (e) {
            console.error('Error setting YouTube volume:', e);
        }
    }
};

window.API = API;
