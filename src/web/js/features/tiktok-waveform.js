// TikTok Range Slider Component for Audio Trimming

/**
 * TikTok range slider with dual handles for start/end time selection
 */
class TikTokRangeSlider {
    // Constants
    static MIN_GAP_SECONDS = 0.1;
    static PERCENT_MULTIPLIER = 100;
    static ELEMENT_IDS = {
        HANDLE_START: 'tt-rs-handle-start',
        HANDLE_END: 'tt-rs-handle-end',
        RANGE: 'tt-rs-range',
        TIME_START: 'tt-rs-time-start',
        TIME_END: 'tt-rs-time-end'
    };

    constructor(containerId, tiktokUrl) {
        this.container = document.getElementById(containerId);

        if (!this.container) {
            return;
        }

        this.tiktokUrl = tiktokUrl;
        this.duration = 0;
        this.startTime = 0;
        this.endTime = 0;

        // Dragging state
        this.isDragging = false;
        this.dragHandle = null; // 'start' or 'end'

        this.init();
    }

    async init() {
        // Create UI first with loading state
        this.createUI();
        this.cacheElements();
        this.setupEvents();

        // Show loading state
        this.showLoading();

        // Load duration asynchronously
        await this.loadDuration();

        // Update UI with actual data
        this.updateUI();
    }

    showLoading() {
        if (this.timeStart) this.timeStart.textContent = 'Loading...';
        if (this.timeEnd) this.timeEnd.textContent = 'Loading...';
    }

    async loadDuration() {
        try {
            this.duration = await API.getTikTokDuration(this.tiktokUrl);

            if (this.duration === 0) {
                // Set a default duration for UI display
                this.duration = 1;
            }

            this.loadTrimSettings();
        } catch (error) {
            // Set default duration on error
            this.duration = 1;
        }
    }

    loadTrimSettings() {
        const trimSettings = AppState.getTikTokTrimSettings(this.tiktokUrl);

        if (trimSettings && (trimSettings.start > 0 || trimSettings.end > 0)) {
            this.startTime = trimSettings.start || 0;
            this.endTime = (trimSettings.end && trimSettings.end > 0) ? trimSettings.end : this.duration;
        } else {
            this.startTime = 0;
            this.endTime = this.duration;
        }
    }

    createUI() {
        this.container.innerHTML = `
            <div class="range-slider-wrapper">
                <div class="range-slider-labels">
                    <span class="range-time-start" id="${TikTokRangeSlider.ELEMENT_IDS.TIME_START}">00:00</span>
                    <span class="range-time-end" id="${TikTokRangeSlider.ELEMENT_IDS.TIME_END}">00:00</span>
                </div>
                <div class="range-slider-container">
                    <div class="range-slider-track"></div>
                    <div class="range-slider-range" id="${TikTokRangeSlider.ELEMENT_IDS.RANGE}"></div>
                    <div class="range-slider-handle range-slider-handle-start" id="${TikTokRangeSlider.ELEMENT_IDS.HANDLE_START}"></div>
                    <div class="range-slider-handle range-slider-handle-end" id="${TikTokRangeSlider.ELEMENT_IDS.HANDLE_END}"></div>
                </div>
            </div>
        `;
    }

    cacheElements() {
        this.handleStart = document.getElementById(TikTokRangeSlider.ELEMENT_IDS.HANDLE_START);
        this.handleEnd = document.getElementById(TikTokRangeSlider.ELEMENT_IDS.HANDLE_END);
        this.range = document.getElementById(TikTokRangeSlider.ELEMENT_IDS.RANGE);
        this.timeStart = document.getElementById(TikTokRangeSlider.ELEMENT_IDS.TIME_START);
        this.timeEnd = document.getElementById(TikTokRangeSlider.ELEMENT_IDS.TIME_END);
        this.sliderContainer = this.container.querySelector('.range-slider-container');
    }

    setupEvents() {
        this.handleStart.addEventListener('mousedown', (e) => this.onHandleMouseDown(e, 'start'));
        this.handleEnd.addEventListener('mousedown', (e) => this.onHandleMouseDown(e, 'end'));
        document.addEventListener('mousemove', (e) => this.onMouseMove(e));
        document.addEventListener('mouseup', () => this.onMouseUp());
    }

    onHandleMouseDown(event, handleType) {
        event.preventDefault();
        this.isDragging = true;
        this.dragHandle = handleType;
    }

    onMouseMove(event) {
        if (!this.isDragging) return;

        const newTime = this.calculateTimeFromMousePosition(event);
        this.updateTimeForHandle(newTime);
        this.updateUI();
    }

    onMouseUp() {
        if (this.isDragging) {
            this.isDragging = false;
            this.dragHandle = null;
            this.saveSettings();
        }
    }

    calculateTimeFromMousePosition(event) {
        const rect = this.sliderContainer.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        return percentage * this.duration;
    }

    updateTimeForHandle(newTime) {
        if (this.dragHandle === 'start') {
            this.startTime = Math.max(0, Math.min(newTime, this.endTime - TikTokRangeSlider.MIN_GAP_SECONDS));
        } else if (this.dragHandle === 'end') {
            this.endTime = Math.min(this.duration, Math.max(newTime, this.startTime + TikTokRangeSlider.MIN_GAP_SECONDS));
        }
    }

    updateUI() {
        if (this.duration === 0 || this.duration === 1) {
            // Show default state
            if (this.timeStart) this.timeStart.textContent = '00:00';
            if (this.timeEnd) this.timeEnd.textContent = '00:00';
            return;
        }

        this.updateTimeLabels();
        this.updateHandlePositions();
        this.updateRangeDisplay();
    }

    updateTimeLabels() {
        this.timeStart.textContent = this.formatTime(this.startTime);
        this.timeEnd.textContent = this.formatTime(this.endTime);
    }

    updateHandlePositions() {
        const startPercent = this.calculatePercentage(this.startTime);
        const endPercent = this.calculatePercentage(this.endTime);

        this.handleStart.style.left = `${startPercent}%`;
        this.handleEnd.style.left = `${endPercent}%`;
    }

    updateRangeDisplay() {
        const startPercent = this.calculatePercentage(this.startTime);
        const endPercent = this.calculatePercentage(this.endTime);

        this.range.style.left = `${startPercent}%`;
        this.range.style.width = `${endPercent - startPercent}%`;
    }

    calculatePercentage(time) {
        return (time / this.duration) * TikTokRangeSlider.PERCENT_MULTIPLIER;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    saveSettings() {
        const endToSave = (this.endTime >= this.duration - TikTokRangeSlider.MIN_GAP_SECONDS) ? 0 : this.endTime;
        AppState.setTikTokTrimSettings(this.tiktokUrl, this.startTime, endToSave);
        SoundEvents.saveSettings();
    }
}

// Export to global scope
window.TikTokRangeSlider = TikTokRangeSlider;
