// API Layer - Eel Communication for Soundboard Pro

const API = {
    /**
     * Internal error handling wrapper for API calls
     * @private
     * @param {Function} apiCall - The async API function to execute
     * @param {*} defaultValue - Default value to return on error
     * @param {string} operationName - Name of the operation for logging
     * @param {boolean} shouldThrow - Whether to rethrow the error
     * @returns {Promise<*>} Result of API call or default value
     */
    async _handleApiCall(apiCall, defaultValue, operationName, shouldThrow = false) {
        try {
            return await apiCall();
        } catch (error) {
            console.error(`[API Error] ${operationName}:`, error);
            if (shouldThrow) throw error;
            return defaultValue;
        }
    },

    // ==================== Settings ====================

    /**
     * Loads application settings from backend
     * @returns {Promise<Object>} Settings object or empty object if failed
     */
    async loadSettings() {
        return this._handleApiCall(
            () => eel.get_settings()(),
            {},
            'loadSettings'
        );
    },

    /**
     * Saves application settings to backend
     * @param {Object} settings - Settings object to save
     * @returns {Promise<void>}
     */
    async saveSettings(settings) {
        return this._handleApiCall(
            () => eel.save_settings(settings)(),
            undefined,
            'saveSettings'
        );
    },

    // ==================== Sounds ====================

    /**
     * Retrieves list of available sounds
     * @returns {Promise<Array<string>>} Array of sound names
     */
    async getSounds() {
        return this._handleApiCall(
            () => eel.get_sounds()(),
            [],
            'getSounds'
        );
    },

    /**
     * Plays a sound with specified volume and pitch
     * @param {string} name - Name of the sound to play
     * @param {number} volume - Volume level (0.0 to 1.0)
     * @param {number} pitch - Pitch multiplier (1.0 = normal)
     * @returns {Promise<void>}
     */
    async playSound(name, volume, pitch) {
        return this._handleApiCall(
            () => eel.play_sound(name, volume, pitch)(),
            undefined,
            `playSound(${name})`
        );
    },

    /**
     * Stops all currently playing sounds
     * @returns {Promise<void>}
     */
    async stopAll() {
        return this._handleApiCall(
            () => eel.stop_all()(),
            undefined,
            'stopAll'
        );
    },

    /**
     * Gets the name of currently playing sound
     * @returns {Promise<string|null>} Sound name or null if nothing playing
     */
    async getPlayingSound() {
        return this._handleApiCall(
            () => eel.get_playing_sound()(),
            null,
            'getPlayingSound'
        );
    },

    /**
     * Opens file dialog to add a new sound
     * @returns {Promise<boolean>} True if sound was added successfully
     */
    async addSoundDialog() {
        return this._handleApiCall(
            () => eel.add_sound_dialog()(),
            false,
            'addSoundDialog'
        );
    },

    /**
     * Adds a sound from base64 encoded data
     * @param {string} filename - Name of the sound file
     * @param {string} base64 - Base64 encoded audio data
     * @returns {Promise<boolean>} True if sound was added successfully
     */
    async addSoundBase64(filename, base64) {
        return this._handleApiCall(
            () => eel.add_sound_base64(filename, base64)(),
            false,
            `addSoundBase64(${filename})`
        );
    },

    /**
     * Deletes a sound from the library
     * @param {string} name - Name of the sound to delete
     * @returns {Promise<void>}
     */
    async deleteSound(name) {
        return this._handleApiCall(
            () => eel.delete_sound(name)(),
            undefined,
            `deleteSound(${name})`
        );
    },

    // ==================== Microphone ====================

    /**
     * Checks if microphone passthrough is enabled
     * @returns {Promise<boolean>} True if mic is enabled
     */
    async isMicEnabled() {
        return this._handleApiCall(
            () => eel.is_mic_enabled()(),
            false,
            'isMicEnabled'
        );
    },

    /**
     * Toggles microphone passthrough on/off
     * @param {boolean} enabled - True to enable, false to disable
     * @returns {Promise<void>}
     * @throws {Error} Rethrows error for UI handling
     */
    async toggleMicPassthrough(enabled) {
        return this._handleApiCall(
            () => eel.toggle_mic_passthrough(enabled)(),
            undefined,
            `toggleMicPassthrough(${enabled})`,
            true // Rethrow error for UI to handle
        );
    },

    /**
     * Sets microphone volume level
     * @param {number} volume - Volume level (0.0 to 3.0)
     * @returns {Promise<void>}
     */
    async setMicVolume(volume) {
        return this._handleApiCall(
            () => eel.set_mic_volume(volume)(),
            undefined,
            `setMicVolume(${volume})`
        );
    },

    // ==================== YouTube ====================

    /**
     * Plays a YouTube video by URL
     * @param {string} url - YouTube video URL
     * @returns {Promise<Object>} Result object with success status and title/error
     */
    async playYoutube(url) {
        return this._handleApiCall(
            () => eel.play_youtube(url)(),
            { success: false, error: 'Failed to connect to backend' },
            `playYoutube(${url})`
        );
    },

    /**
     * Stops currently playing YouTube video
     * @returns {Promise<void>}
     */
    async stopYoutube() {
        return this._handleApiCall(
            () => eel.stop_youtube()(),
            undefined,
            'stopYoutube'
        );
    },

    /**
     * Pauses currently playing YouTube video
     * @returns {Promise<void>}
     */
    async pauseYoutube() {
        return this._handleApiCall(
            () => eel.pause_youtube()(),
            undefined,
            'pauseYoutube'
        );
    },

    /**
     * Resumes paused YouTube video
     * @returns {Promise<void>}
     */
    async resumeYoutube() {
        return this._handleApiCall(
            () => eel.resume_youtube()(),
            undefined,
            'resumeYoutube'
        );
    },

    /**
     * Gets current YouTube playback information
     * @returns {Promise<Object>} Info object with playing status, title, url, and paused state
     */
    async getYoutubeInfo() {
        return this._handleApiCall(
            () => eel.get_youtube_info()(),
            { playing: false },
            'getYoutubeInfo'
        );
    },

    /**
     * Sets YouTube playback volume
     * @param {number} volume - Volume level (0.0 to 1.0)
     * @returns {Promise<void>}
     */
    async setYoutubeVolume(volume) {
        return this._handleApiCall(
            () => eel.set_youtube_volume(volume)(),
            undefined,
            `setYoutubeVolume(${volume})`
        );
    },

    /**
     * Downloads and saves YouTube video as a sound file
     * @param {string} url - YouTube video URL
     * @returns {Promise<Object>} Result object with success status and name/error
     */
    async saveYoutubeAsSound(url) {
        return this._handleApiCall(
            () => eel.save_youtube_as_sound(url)(),
            { success: false, error: 'Failed to save' },
            `saveYoutubeAsSound(${url})`
        );
    },

    /**
     * Retrieves list of saved YouTube items
     * @returns {Promise<Array<Object>>} Array of YouTube item objects
     */
    async getYoutubeItems() {
        return this._handleApiCall(
            () => eel.get_youtube_items()(),
            [],
            'getYoutubeItems'
        );
    },

    /**
     * Adds a new YouTube item to the library
     * @param {string} url - YouTube video URL
     * @returns {Promise<Object>} Result object with success status and title/error
     */
    async addYoutubeItem(url) {
        return this._handleApiCall(
            () => eel.add_youtube_item(url)(),
            { success: false, error: 'Failed to add item' },
            `addYoutubeItem(${url})`
        );
    },

    /**
     * Deletes a YouTube item from the library
     * @param {string} url - YouTube video URL
     * @returns {Promise<Object>} Result object with success status
     */
    async deleteYoutubeItem(url) {
        return this._handleApiCall(
            () => eel.delete_youtube_item(url)(),
            { success: false, error: 'Failed to delete item' },
            `deleteYoutubeItem(${url})`
        );
    }
};

window.API = API;
