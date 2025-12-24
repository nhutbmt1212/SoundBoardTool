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
     * @param {number} startTime - Start time in seconds (0 = from beginning)
     * @param {number} endTime - End time in seconds (0 = to end)
     * @returns {Promise<void>}
     */
    async playSound(name, volume, pitch, startTime = 0, endTime = 0, loop = false) {
        return this._handleApiCall(
            () => eel.play_sound(name, volume, pitch, startTime, endTime, loop)(),
            undefined,
            `playSound(${name})`
        );
    },

    /**
     * Sets sound playback volume (for live updates)
     * @param {number} volume - Volume level (0.0 to 1.0) or higher for scream
     * @returns {Promise<void>}
     */
    async setSoundVolume(volume) {
        return this._handleApiCall(
            () => eel.set_volume(volume)(),
            undefined,
            `setSoundVolume(${volume})`
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

    async pauseSound() {
        return this._handleApiCall(
            () => eel.pause_sound()(),
            undefined,
            'pauseSound'
        );
    },

    async resumeSound() {
        return this._handleApiCall(
            () => eel.resume_sound()(),
            undefined,
            'resumeSound'
        );
    },

    async isSoundPaused() {
        return this._handleApiCall(
            () => eel.is_sound_paused()(),
            false,
            'isSoundPaused'
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

    /**
     * Gets audio file duration in seconds
     * @param {string} name - Name of the sound
     * @returns {Promise<number>} Duration in seconds
     */
    async getAudioDuration(name) {
        return this._handleApiCall(
            () => eel.get_audio_duration(name)(),
            0,
            `getAudioDuration(${name})`
        );
    },

    /**
     * Gets waveform data for visualization
     * @param {string} name - Name of the sound
     * @param {number} samples - Number of samples to return (default 200)
     * @returns {Promise<Array<number>>} Array of amplitude values
     */
    async getWaveformData(name, samples = 200) {
        return this._handleApiCall(
            () => eel.get_waveform_data(name, samples)(),
            [],
            `getWaveformData(${name})`
        );
    },

    /**
     * Set sound loop state
     * @param {boolean} enabled - Loop state
     */
    async setSoundLoop(enabled) {
        return this._handleApiCall(
            () => eel.set_sound_loop(enabled)(),
            undefined,
            `setSoundLoop(${enabled})`
        );
    },

    /**
     * Get sound loop state
     * @returns {Promise<boolean>} Loop state
     */
    async getSoundLoop() {
        return this._handleApiCall(
            () => eel.get_sound_loop()(),
            false,
            'getSoundLoop'
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
     * @param {number} volume - Volume level (0.0 to 1.0)
     * @param {number} pitch - Pitch multiplier (1.0 = normal)
     * @param {number} startTime - Start time/trim in seconds
     * @param {number} endTime - End time/trim in seconds
     * @returns {Promise<Object>} Result object with success status and title/error
     */
    async playYoutube(url, volume = 1.0, pitch = 1.0, startTime = 0, endTime = 0, loop = false) {
        return this._handleApiCall(
            () => eel.play_youtube(url, volume, pitch, startTime, endTime, loop)(),
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
    },

    /**
     * Gets YouTube video duration in seconds
     * @param {string} url - YouTube video URL
     * @returns {Promise<number>} Duration in seconds
     */
    async getYoutubeDuration(url) {
        return this._handleApiCall(
            () => eel.get_youtube_duration(url)(),
            0,
            `getYoutubeDuration(${url})`
        );
    },

    /**
     * Set YouTube loop state
     * @param {boolean} enabled - Loop state
     */
    async setYoutubeLoop(enabled) {
        return this._handleApiCall(
            () => eel.set_youtube_loop(enabled)(),
            undefined,
            `setYoutubeLoop(${enabled})`
        );
    },

    /**
     * Get YouTube loop state
     * @returns {Promise<boolean>} Loop state
     */
    async getYoutubeLoop() {
        return this._handleApiCall(
            () => eel.get_youtube_loop()(),
            false,
            'getYoutubeLoop'
        );
    },

    // ==================== TikTok ====================

    /**
     * Plays a TikTok video by URL
     * @param {string} url - TikTok video URL
     * @param {number} volume - Volume level (0.0 to 1.0)
     * @param {number} pitch - Pitch multiplier (1.0 = normal)
     * @param {number} startTime - Start time/trim in seconds
     * @param {number} endTime - End time/trim in seconds
     * @returns {Promise<Object>} Result object with success status and title/error
     */
    async playTikTok(url, volume = 1.0, pitch = 1.0, startTime = 0, endTime = 0, loop = false) {
        return this._handleApiCall(
            () => eel.play_tiktok(url, volume, pitch, startTime, endTime, loop)(),
            { success: false, error: 'Failed to connect to backend' },
            `playTikTok(${url})`
        );
    },

    /**
     * Stops currently playing TikTok video
     * @returns {Promise<void>}
     */
    async stopTikTok() {
        return this._handleApiCall(
            () => eel.stop_tiktok()(),
            undefined,
            'stopTikTok'
        );
    },

    /**
     * Pauses currently playing TikTok video
     * @returns {Promise<void>}
     */
    async pauseTikTok() {
        return this._handleApiCall(
            () => eel.pause_tiktok()(),
            undefined,
            'pauseTikTok'
        );
    },

    /**
     * Resumes paused TikTok video
     * @returns {Promise<void>}
     */
    async resumeTikTok() {
        return this._handleApiCall(
            () => eel.resume_tiktok()(),
            undefined,
            'resumeTikTok'
        );
    },

    /**
     * Gets current TikTok playback information
     * @returns {Promise<Object>} Info object with playing status, title, url, and paused state
     */
    async getTikTokInfo() {
        return this._handleApiCall(
            () => eel.get_tiktok_info()(),
            { playing: false },
            'getTikTokInfo'
        );
    },

    /**
     * Sets TikTok playback volume
     * @param {number} volume - Volume level (0.0 to 1.0)
     * @returns {Promise<void>}
     */
    async setTikTokVolume(volume) {
        return this._handleApiCall(
            () => eel.set_tiktok_volume(volume)(),
            undefined,
            `setTikTokVolume(${volume})`
        );
    },

    /**
     * Downloads and saves TikTok video as a sound file
     * @param {string} url - TikTok video URL
     * @returns {Promise<Object>} Result object with success status and name/error
     */
    async saveTikTokAsSound(url) {
        return this._handleApiCall(
            () => eel.save_tiktok_as_sound(url)(),
            { success: false, error: 'Failed to save' },
            `saveTikTokAsSound(${url})`
        );
    },

    /**
     * Retrieves list of saved TikTok items
     * @returns {Promise<Array<Object>>} Array of TikTok item objects
     */
    async getTikTokItems() {
        return this._handleApiCall(
            () => eel.get_tiktok_items()(),
            [],
            'getTikTokItems'
        );
    },

    /**
     * Adds a new TikTok item to the library
     * @param {string} url - TikTok video URL
     * @returns {Promise<Object>} Result object with success status and title/error
     */
    async addTikTokItem(url) {
        return this._handleApiCall(
            () => eel.add_tiktok_item(url)(),
            { success: false, error: 'Failed to add item' },
            `addTikTokItem(${url})`
        );
    },

    /**
     * Deletes a TikTok item from the library
     * @param {string} url - TikTok video URL
     * @returns {Promise<Object>} Result object with success status
     */
    async deleteTikTokItem(url) {
        return this._handleApiCall(
            () => eel.delete_tiktok_item(url)(),
            { success: false, error: 'Failed to delete item' },
            `deleteTikTokItem(${url})`
        );
    },

    /**
     * Gets TikTok video duration in seconds
     * @param {string} url - TikTok video URL
     * @returns {Promise<number>} Duration in seconds
     */
    async getTikTokDuration(url) {
        return this._handleApiCall(
            () => eel.get_tiktok_duration(url)(),
            0,
            `getTikTokDuration(${url})`
        );
    },
    /**
     * Set TikTok loop state
     * @param {boolean} enabled - Loop state
     */
    async setTikTokLoop(enabled) {
        return this._handleApiCall(
            () => eel.set_tiktok_loop(enabled)(),
            undefined,
            `setTikTokLoop(${enabled})`
        );
    },

    /**
     * Get TikTok loop state
     * @returns {Promise<boolean>} Loop state
     */
    async getTikTokLoop() {
        return this._handleApiCall(
            () => eel.get_tiktok_loop()(),
            false,
            'getTikTokLoop'
        );
    },

    // ==================== TTS (Text-to-Speech) ====================

    /**
     * Generate speech from text and play
     * @param {string} text - Text to convert to speech
     * @param {string} voice - Edge TTS voice name
     * @param {number} volume - Volume level (0.0 to 1.0)
     * @returns {Promise<Object>} Result object with success status
     */
    async generateAndPlayTTS(text, voice, volume) {
        return this._handleApiCall(
            () => eel.generate_and_play_tts(text, voice, volume)(),
            { success: false, error: 'Failed to generate speech' },
            `generateAndPlayTTS`
        );
    },

    /**
     * Get available TTS voices
     * @returns {Promise<Object>} Voice ID to display name mapping
     */
    async getTTSVoices() {
        return this._handleApiCall(
            () => eel.get_tts_voices()(),
            {},
            'getTTSVoices'
        );
    },

    /**
     * Stop TTS playback
     * @returns {Promise<void>}
     */
    async stopTTS() {
        return this._handleApiCall(
            () => eel.stop_tts()(),
            undefined,
            'stopTTS'
        );
    },

    /**
     * Cancel TTS generation
     * @returns {Promise<Object>}
     */
    async cancelTTS() {
        return this._handleApiCall(
            () => eel.cancel_tts()(),
            { success: false },
            'cancelTTS'
        );
    },

    /**
     * Check if TTS is currently generating
     * @returns {Promise<boolean>}
     */
    async isTTSGenerating() {
        return this._handleApiCall(
            () => eel.is_tts_generating()(),
            false,
            'isTTSGenerating'
        );
    }
};

window.API = API;
