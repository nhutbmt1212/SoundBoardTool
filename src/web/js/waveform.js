// Range Slider Component for Audio Trimming

/**
 * Simple range slider with dual handles for start/end time selection
 */
class RangeSlider {
    // Constants
    static MIN_GAP_SECONDS = 0.1;
    static PERCENT_MULTIPLIER = 100;
    static ELEMENT_IDS = {
        HANDLE_START: 'rs-handle-start',
        HANDLE_END: 'rs-handle-end',
        RANGE: 'rs-range',
        TIME_START: 'rs-time-start',
        TIME_END: 'rs-time-end'
    };

    constructor(containerId, soundName) {
        console.log('[RangeSlider] Initializing for:', soundName);
        this.container = document.getElementById(containerId);

        if (!this.container) {
            console.error('[RangeSlider] Container not found:', containerId);
            return;
        }

        this.soundName = soundName;
        this.duration = 0;
        this.startTime = 0;
        this.endTime = 0;

        // Dragging state
        this.isDragging = false;
        this.dragHandle = null; // 'start' or 'end'

        this.init();
    }

    async init() {
        await this.loadDuration();
        this.createUI();
        this.cacheElements();
        this.setupEvents();
        this.updateUI();

        console.log('[RangeSlider] Initialized');
    }

    async loadDuration() {
        try {
            this.duration = await API.getAudioDuration(this.soundName);
            console.log('[RangeSlider] Duration:', this.duration);

            this.loadTrimSettings();
        } catch (error) {
            console.error('[RangeSlider] Error loading duration:', error);
        }
    }

    loadTrimSettings() {
        const trimSettings = AppState.getTrimSettings(this.soundName);

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
                    <span class="range-time-start" id="${RangeSlider.ELEMENT_IDS.TIME_START}">00:00</span>
                    <span class="range-time-end" id="${RangeSlider.ELEMENT_IDS.TIME_END}">00:00</span>
                </div>
                <div class="range-slider-container">
                    <div class="range-slider-track"></div>
                    <div class="range-slider-range" id="${RangeSlider.ELEMENT_IDS.RANGE}"></div>
                    <div class="range-slider-handle range-slider-handle-start" id="${RangeSlider.ELEMENT_IDS.HANDLE_START}"></div>
                    <div class="range-slider-handle range-slider-handle-end" id="${RangeSlider.ELEMENT_IDS.HANDLE_END}"></div>
                </div>
            </div>
        `;
    }

    cacheElements() {
        this.handleStart = document.getElementById(RangeSlider.ELEMENT_IDS.HANDLE_START);
        this.handleEnd = document.getElementById(RangeSlider.ELEMENT_IDS.HANDLE_END);
        this.range = document.getElementById(RangeSlider.ELEMENT_IDS.RANGE);
        this.timeStart = document.getElementById(RangeSlider.ELEMENT_IDS.TIME_START);
        this.timeEnd = document.getElementById(RangeSlider.ELEMENT_IDS.TIME_END);
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
            this.startTime = Math.max(0, Math.min(newTime, this.endTime - RangeSlider.MIN_GAP_SECONDS));
        } else if (this.dragHandle === 'end') {
            this.endTime = Math.min(this.duration, Math.max(newTime, this.startTime + RangeSlider.MIN_GAP_SECONDS));
        }
    }

    updateUI() {
        if (this.duration === 0) return;

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
        return (time / this.duration) * RangeSlider.PERCENT_MULTIPLIER;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    saveSettings() {
        const endToSave = (this.endTime >= this.duration - RangeSlider.MIN_GAP_SECONDS) ? 0 : this.endTime;
        console.log('[RangeSlider] Saving:', this.startTime, 'to', endToSave);
        AppState.setTrimSettings(this.soundName, this.startTime, endToSave);
        SoundEvents.saveSettings();
    }
}

// Export to global scope
window.RangeSlider = RangeSlider;
