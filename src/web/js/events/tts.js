// TTS Event Handlers for Soundboard Pro - Text-to-Speech using Edge TTS

/**
 * TTS event handlers module
 * Handles all TTS-related user interactions and playback
 */
const TTSEvents = {
    // ==================== Playback Control ====================

    /**
     * Generate speech from text and play through VB-Cable
     * @returns {Promise<void>}
     */
    async generateAndPlay() {
        const textInput = document.getElementById('tts-text');
        const voiceSelect = document.getElementById('tts-voice');
        const volumeSlider = document.getElementById('tts-volume');
        const generateBtn = document.getElementById('tts-generate-btn');
        const statusEl = document.getElementById('tts-status');

        const text = textInput.value.trim();
        const voice = voiceSelect.value;
        const volume = parseInt(volumeSlider.value) / 100;

        if (!text) {
            Notifications.warning('Vui lòng nhập văn bản');
            return;
        }

        // Show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="btn-loading"></span> Đang tạo...';
        statusEl.style.display = 'block';
        statusEl.textContent = 'Đang tạo giọng nói...';
        statusEl.className = 'tts-status loading';

        try {
            const result = await API.generateAndPlayTTS(text, voice, volume);

            if (result.success) {
                statusEl.textContent = 'Đang phát...';
                statusEl.className = 'tts-status playing';
                Notifications.success('Đang phát giọng nói');
            } else {
                statusEl.textContent = result.error || 'Lỗi tạo giọng nói';
                statusEl.className = 'tts-status error';
                Notifications.error(result.error || 'Lỗi tạo giọng nói');
            }
        } catch (e) {
            console.error('[TTS] Error:', e);
            statusEl.textContent = 'Lỗi kết nối';
            statusEl.className = 'tts-status error';
            Notifications.error('Lỗi kết nối');
        } finally {
            // Reset button state
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<span id="icon-tts-play"></span> Phát giọng nói';
            this.setupIcons();
        }
    },

    /**
     * Stop TTS playback
     * @returns {Promise<void>}
     */
    async stop() {
        const statusEl = document.getElementById('tts-status');

        try {
            await API.stopTTS();
            statusEl.style.display = 'none';
            Notifications.info('Đã dừng');
        } catch (e) {
            console.error('[TTS] Stop error:', e);
        }
    },

    // ==================== Settings ====================

    /**
     * Handle volume slider change
     * @param {number} value - Volume value (0-100)
     */
    onVolumeChange(value) {
        const volumeLabel = document.getElementById('tts-volume-value');
        if (volumeLabel) {
            volumeLabel.textContent = `${value}%`;
        }
    },

    // ==================== UI Setup ====================

    /**
     * Initialize TTS tab icons
     */
    setupIcons() {
        // Tab icon
        const tabIcon = document.getElementById('icon-tab-tts');
        if (tabIcon && typeof IconManager !== 'undefined') {
            tabIcon.innerHTML = IconManager.get('mic', { size: 20 });
        }

        // Button icons
        const playIcon = document.getElementById('icon-tts-play');
        if (playIcon && typeof IconManager !== 'undefined') {
            playIcon.innerHTML = IconManager.get('play', { size: 18 });
        }

        const stopIcon = document.getElementById('icon-tts-stop');
        if (stopIcon && typeof IconManager !== 'undefined') {
            stopIcon.innerHTML = IconManager.get('stop', { size: 18 });
        }
    },

    /**
     * Initialize TTS module
     */
    init() {
        this.setupIcons();
    }
};

// Export to global scope
window.TTSEvents = TTSEvents;
