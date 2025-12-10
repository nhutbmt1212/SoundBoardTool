// UI Helpers - Common HTML Building Blocks
// Extracted from ui.js to eliminate duplication

const UIHelpers = {
    /**
     * Build panel header with editable name input
     * @param {Object} config - {id, displayName, placeholder, onChangeHandler, infoText}
     */
    buildPanelHeader(config) {
        return `
            <div class="panel-header">
                <input type="text" class="panel-sound-name editable" id="${config.id}" 
                       value="${Utils.escapeAttr(config.displayName)}" 
                       placeholder="${Utils.escapeAttr(config.placeholder)}"
                       onchange="${config.onChangeHandler}(this.value)"
                       onkeydown="if(event.key==='Enter')this.blur()">
                <div class="panel-sound-info" title="${Utils.escapeAttr(config.infoText)}">${Utils.escapeHtml(config.infoText)}</div>
            </div>
        `;
    },

    /**
     * Build preview waveform animation
     * @param {boolean} isActive - Whether to show active state
     */
    buildPreviewWave(isActive = false) {
        return `
            <div class="panel-preview">
                <div class="preview-wave ${isActive ? 'scream-active' : ''}">
                    <div class="wave-animation">
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                    </div>
                </div>
            </div>
        `;
    },

    /**
     * Build keybind input section
     * @param {Object} config - {inputId, value, onClickHandler}
     */
    buildKeybindSection(config) {
        return `
            <div class="panel-section">
                <div class="panel-section-title">${icon('keyboard', 16)} Keybind</div>
                <input type="text" class="keybind-input" id="${config.inputId}" 
                       value="${config.value}" 
                       placeholder="Click to set keybind"
                       readonly
                       onclick="${config.onClickHandler}()">
            </div>
        `;
    },

    /**
     * Build volume control section
     * @param {Object} config - {sliderId, valueId, volume, title, onInputHandler, onChangeHandler}
     */
    buildVolumeSection(config) {
        return `
            <div class="panel-section">
                <div class="panel-section-title">${icon('volume', 16)} ${config.title}</div>
                <div class="volume-control">
                    <input type="range" class="volume-slider" id="${config.sliderId}" 
                           min="0" max="100" value="${config.volume}"
                           oninput="${config.onInputHandler}(this.value)"
                           onchange="${config.onChangeHandler}(this.value)">
                    <span class="volume-value" id="${config.valueId}">${config.volume}${config.showPercent !== false ? '%' : ''}</span>
                </div>
            </div>
        `;
    },

    /**
     * Build trim audio section
     * @param {string} containerId - Container element ID
     */
    buildTrimSection(containerId) {
        return `
            <div class="panel-section">
                <div class="panel-section-title">${icon('clock', 16)} Trim Audio</div>
                <div id="${containerId}"></div>
            </div>
        `;
    },

    /**
     * Build scream mode toggle
     * @param {Object} config - {checkboxId, isActive, onChangeHandler, url (optional)}
     */
    buildScreamToggle(config) {
        const onChangeAttr = config.url
            ? `onchange="${config.onChangeHandler}('${Utils.escapeAttr(config.url)}')"`
            : `onchange="${config.onChangeHandler}()"`;

        return `
            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('scream', { size: 16 })} Scream Mode</div>
                <label class="scream-toggle">
                    <input type="checkbox" id="${config.checkboxId}" ${config.isActive ? 'checked' : ''} ${onChangeAttr}>
                    <span class="scream-slider"></span>
                    <span class="scream-label">${config.isActive ? 'ON - 5000% BOOST!' : 'OFF'}</span>
                </label>
                <div class="scream-hint">Boost volume to max for trolling</div>
            </div>
        `;
    },

    /**
     * Build chipmunk mode toggle
     * @param {Object} config - {checkboxId, isActive, onChangeHandler, url (optional)}
     */
    buildChipmunkToggle(config) {
        const onChangeAttr = config.url
            ? `onchange="${config.onChangeHandler}('${Utils.escapeAttr(config.url)}')"`
            : `onchange="${config.onChangeHandler}()"`;

        return `
            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('chipmunk', { size: 16 })} Chipmunk Mode</div>
                <label class="pitch-toggle">
                    <input type="checkbox" id="${config.checkboxId}" ${config.isActive ? 'checked' : ''} ${onChangeAttr}>
                    <span class="pitch-slider"></span>
                    <span class="pitch-label">${config.isActive ? 'ON - HIGH PITCH!' : 'OFF'}</span>
                </label>
                <div class="pitch-hint">Speed up audio for chipmunk voice</div>
            </div>
        `;
    },

    /**
     * Build empty state placeholder
     * @param {Object} config - {iconName, title, message}
     */
    buildEmptyState(config) {
        return `
            <div class="empty-state">
                <span class="empty-icon">${IconManager.get(config.iconName, { size: 64 })}</span>
                <h2>${Utils.escapeHtml(config.title)}</h2>
                <p>${config.message}</p>
            </div>
        `;
    },

    /**
     * Build loading card for downloads
     * @param {Object} config - {url, className}
     */
    buildLoadingCard(config) {
        return `
            <div class="sound-thumbnail youtube-thumbnail">
                <div class="loading-pie" style="--p: 0;"></div>
                <div class="loading-percent">0%</div>
            </div>
            <div class="sound-name">Downloading...</div>
            <div class="sound-info" style="font-size: 10px; color: var(--text-muted); padding: 0 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${Utils.escapeHtml(config.url)}</div>
        `;
    }
};

window.UIHelpers = UIHelpers;
