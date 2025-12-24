// TTS Event Handlers for Soundboard Pro - Text-to-Speech using Edge TTS

/**
 * TTS event handlers module
 * Handles all TTS-related user interactions and playback
 */
const TTSEvents = {
    _isGenerating: false,

    // ==================== Playback Control ====================

    /**
     * Generate speech from text and play through VB-Cable
     * Or cancel if already generating
     * @returns {Promise<void>}
     */
    async generateAndPlay() {
        const generateBtn = document.getElementById('tts-generate-btn');
        const statusEl = document.getElementById('tts-status');

        // If generating, cancel it
        if (this._isGenerating) {
            console.log('[TTS] Cancel requested by user');
            this._isGenerating = false;
            API.cancelTTS();  // Don't await - fire and forget
            this._resetButton();
            statusEl.textContent = 'Đã hủy';
            statusEl.className = 'tts-status error';
            statusEl.style.display = 'block';
            Notifications.info('Đã hủy tạo giọng nói');
            return;
        }

        const textInput = document.getElementById('tts-text');
        const voiceSelect = document.getElementById('tts-voice');
        const volumeSlider = document.getElementById('tts-volume');

        const text = textInput.value.trim();
        const voice = voiceSelect.value;
        const volume = parseInt(volumeSlider.value) / 100;

        if (!text) {
            Notifications.warning('Vui lòng nhập văn bản');
            return;
        }

        // Show loading state with cancel option
        this._isGenerating = true;
        generateBtn.innerHTML = '<span class="btn-loading"></span> Nhấn để hủy';
        generateBtn.classList.add('generating');
        statusEl.style.display = 'block';
        statusEl.textContent = 'Đang tạo giọng nói...';
        statusEl.className = 'tts-status loading';

        // Run generation async - don't block UI
        this._runGeneration(text, voice, volume);
    },

    /**
     * Run TTS generation asynchronously
     */
    async _runGeneration(text, voice, volume) {
        const statusEl = document.getElementById('tts-status');

        try {
            const result = await API.generateAndPlayTTS(text, voice, volume);

            // Check if cancelled during generation
            if (!this._isGenerating) {
                return; // Was cancelled
            }

            if (result.cancelled) {
                return; // Backend returned cancelled
            }

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
            if (!this._isGenerating) return; // Was cancelled
            console.error('[TTS] Error:', e);
            statusEl.textContent = 'Lỗi kết nối';
            statusEl.className = 'tts-status error';
            Notifications.error('Lỗi kết nối');
        } finally {
            this._isGenerating = false;
            this._resetButton();
        }
    },

    /**
     * Reset generate button to default state
     */
    _resetButton() {
        const generateBtn = document.getElementById('tts-generate-btn');
        if (generateBtn) {
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
            this._isGenerating = false;
            await API.stopTTS();
            this._resetButton();
            statusEl.style.display = 'none';
            Notifications.info('Đã dừng');
        } catch (e) {
            console.error('[TTS] Stop error:', e);
        }
    },

    // ==================== Settings ====================

    /**
     * Handle volume slider change
     * @param {number} value - Volume value (0-200)
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
        this._isGenerating = false;
        this._resetButton();
        this.setupIcons();
    }
};

// Export to global scope
window.TTSEvents = TTSEvents;

