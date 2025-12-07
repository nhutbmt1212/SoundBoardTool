// YouTube Range Slider Component for Audio Trimming

/**
 * YouTube range slider with dual handles for start/end time selection
 */
class YouTubeRangeSlider {
    // Constants
    static MIN_GAP_SECONDS = 0.1;
    static PERCENT_MULTIPLIER = 100;
    static ELEMENT_IDS = {
        HANDLE_START: 'yt-rs-handle-start',
        HANDLE_END: 'yt-rs-handle-end',
        RANGE: 'yt-rs-range',
        TIME_START: 'yt-rs-time-start',
        TIME_END: 'yt-rs-time-end'
    };

    constructor(containerId, youtubeUrl) {
        console.log('[YouTubeRangeSlider] Initializing for:', youtubeUrl);
        this.container = document.getElementById(containerId);

        if (!this.container) {
            console.error('[YouTubeRangeSlider] Container not found:', containerId);
            return;
        }

        this.youtubeUrl = youtubeUrl;
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

        console.log('[YouTubeRangeSlider] Initialized with duration:', this.duration);
    }
    
    showLoading() {
        if (this.timeStart) this.timeStart.textContent = 'Loading...';
        if (this.timeEnd) this.timeEnd.textContent = 'Loading...';
    }

    async loadDuration() {
        try {
            this.duration = await API.getYoutubeDuration(this.youtubeUrl);
            console.log('[YouTubeRangeSlider] Duration loaded:', this.duration);

            if (this.duration === 0) {
                console.warn('[YouTubeRangeSlider] Duration is 0, video may not be cached yet');
                // Set a default duration for UI display
                this.duration = 1;
            }

            this.loadTrimSettings();
        } catch (error) {
            console.error('[YouTubeRangeSlider] Error loading duration:', error);
            // Set default duration on error
            this.duration = 1;
        }
    }

    loadTrimSettings() {
        const trimSettings = AppState.getYoutubeTrimSettings(this.youtubeUrl);

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
                    <span class="range-time-start" id="${YouTubeRangeSlider.ELEMENT_IDS.TIME_START}">00:00</span>
                    <span class="range-time-end" id="${YouTubeRangeSlider.ELEMENT_IDS.TIME_END}">00:00</span>
                </div>
                <div class="range-slider-container">
                    <div class="range-slider-track"></div>
                    <div class="range-slider-range" id="${YouTubeRangeSlider.ELEMENT_IDS.RANGE}"></div>
                    <div class="range-slider-handle range-slider-handle-start" id="${YouTubeRangeSlider.ELEMENT_IDS.HANDLE_START}"></div>
                    <div class="range-slider-handle range-slider-handle-end" id="${YouTubeRangeSlider.ELEMENT_IDS.HANDLE_END}"></div>
                </div>
            </div>
        `;
    }

    cacheElements() {
        this.handleStart = document.getElementById(YouTubeRangeSlider.ELEMENT_IDS.HANDLE_START);
        this.handleEnd = document.getElementById(YouTubeRangeSlider.ELEMENT_IDS.HANDLE_END);
        this.range = document.getElementById(YouTubeRangeSlider.ELEMENT_IDS.RANGE);
        this.timeStart = document.getElementById(YouTubeRangeSlider.ELEMENT_IDS.TIME_START);
        this.timeEnd = document.getElementById(YouTubeRangeSlider.ELEMENT_IDS.TIME_END);
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
            this.startTime = Math.max(0, Math.min(newTime, this.endTime - YouTubeRangeSlider.MIN_GAP_SECONDS));
        } else if (this.dragHandle === 'end') {
            this.endTime = Math.min(this.duration, Math.max(newTime, this.startTime + YouTubeRangeSlider.MIN_GAP_SECONDS));
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
        return (time / this.duration) * YouTubeRangeSlider.PERCENT_MULTIPLIER;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    saveSettings() {
        const endToSave = (this.endTime >= this.duration - YouTubeRangeSlider.MIN_GAP_SECONDS) ? 0 : this.endTime;
        console.log('[YouTubeRangeSlider] Saving:', this.startTime, 'to', endToSave);
        AppState.setYoutubeTrimSettings(this.youtubeUrl, this.startTime, endToSave);
        SoundEvents.saveSettings();
    }
}

// Export to global scope
window.YouTubeRangeSlider = YouTubeRangeSlider;
