// Panel Renderer - Unified panel rendering for all media types
// Eliminates massive duplication between Sound, YouTube, and TikTok panels

const PanelRenderer = {
    /**
     * Render sound panel
     * @param {string} name - Sound file name
     */
    renderSoundPanel(name) {
        const panel = document.getElementById('right-panel');
        const volume = AppState.getVolume(name);
        const keybind = AppState.getKeybind(name);
        const isScream = AppState.isScreamMode(name);
        const isPitch = AppState.isPitchMode(name);
        const displayName = AppState.getDisplayName(name);

        panel.innerHTML = `
            ${UIHelpers.buildPanelHeader({
            id: 'sound-name-input',
            displayName: displayName,
            placeholder: name,
            onChangeHandler: 'EventHandlers.onSoundNameChange',
            infoText: name
        })}
            
            ${UIHelpers.buildPreviewWave(isScream)}
            
            ${UIHelpers.buildKeybindSection({
            inputId: 'keybind-input',
            value: keybind,
            onClickHandler: 'EventHandlers.startKeybindRecordPanel'
        })}
            
            ${UIHelpers.buildVolumeSection({
            sliderId: 'sound-volume',
            valueId: 'volume-value',
            volume: volume,
            title: 'Volume',
            onInputHandler: 'SoundEvents.onVolumeLive',
            onChangeHandler: 'SoundEvents.onVolumeSave',
            showPercent: false
        })}
            
            ${UIHelpers.buildTrimSection('waveform-container')}
            
            ${UIHelpers.buildScreamToggle({
            checkboxId: 'scream-checkbox',
            isActive: isScream,
            onChangeHandler: 'EventHandlers.toggleScreamMode'
        })}
            
            ${UIHelpers.buildChipmunkToggle({
            checkboxId: 'pitch-checkbox',
            isActive: isPitch,
            onChangeHandler: 'EventHandlers.togglePitchMode'
        })}

            ${UIHelpers.buildLoopToggle({
            checkboxId: 'loop-checkbox',
            isActive: AppState.isSoundLoop(name),
            onChangeHandler: 'EventHandlers.toggleLoop'
        })}
            
            ${EffectsHelpers.renderEffectsSection(AppState.getSoundEffects(name))}
            
            <div class="panel-actions">
                <button class="btn-panel btn-play" onclick="EventHandlers.playSound('${Utils.escapeAttr(name)}')">${icon('play', 14)} Play</button>
                <button class="btn-panel btn-stop" onclick="EventHandlers.stopAll()">${icon('stop', 14)} Stop</button>
            </div>
            
            <div class="panel-actions">
                <button class="btn-panel btn-delete" onclick="EventHandlers.deleteSound('${Utils.escapeAttr(name)}')">${icon('trash', 14)} Delete</button>
            </div>
        `;

        // Initialize waveform visualizer after DOM is ready
        setTimeout(() => {
            if (window.RangeSlider) {
                new RangeSlider('waveform-container', name);
            }
            // Setup effects event listeners
            EffectsEvents.setupEffectsListeners('sound', name, AppState.getSoundEffects(name));
        }, 100);
    },

    /**
     * Render YouTube panel
     * @param {Object} item - YouTube item {url, title}
     */
    renderYoutubePanel(item) {
        const panel = document.getElementById('right-panel');
        const displayName = AppState.getYoutubeDisplayName(item.url, item.title);

        panel.innerHTML = `
            ${UIHelpers.buildPanelHeader({
            id: 'yt-name-input',
            displayName: displayName,
            placeholder: item.title,
            onChangeHandler: 'EventHandlers.onYoutubeNameChange',
            infoText: item.url
        })}
            
            ${UIHelpers.buildPreviewWave(true)}
            
            ${UIHelpers.buildKeybindSection({
            inputId: 'yt-keybind-input',
            value: AppState.getYoutubeKeybind(item.url),
            onClickHandler: 'EventHandlers.startYoutubeKeybindRecording'
        })}
            
            ${UIHelpers.buildVolumeSection({
            sliderId: 'youtube-volume',
            valueId: 'youtube-volume-value',
            volume: AppState.getYoutubeVolume(item.url),
            title: 'Global YouTube Volume',
            onInputHandler: 'YouTubeEvents.onVolumeLive',
            onChangeHandler: 'YouTubeEvents.onVolumeSave'
        })}
            
            ${UIHelpers.buildTrimSection('youtube-waveform-container')}

            ${UIHelpers.buildScreamToggle({
            checkboxId: 'yt-scream-checkbox',
            isActive: AppState.isYoutubeScreamMode(item.url),
            onChangeHandler: 'EventHandlers.toggleYoutubeScreamMode',
            url: item.url
        })}
            
            ${UIHelpers.buildChipmunkToggle({
            checkboxId: 'yt-pitch-checkbox',
            isActive: AppState.isYoutubePitchMode(item.url),
            onChangeHandler: 'EventHandlers.toggleYoutubePitchMode',
            url: item.url
        })}

            ${UIHelpers.buildLoopToggle({
            checkboxId: 'yt-loop-checkbox',
            isActive: AppState.isYoutubeLoop(item.url),
            onChangeHandler: 'EventHandlers.toggleYoutubeLoop',
            url: item.url
        })}
            
            ${EffectsHelpers.renderEffectsSection(AppState.getYoutubeEffects(item.url))}
            
            <div class="panel-actions">
                <button class="btn-panel btn-play" onclick="playYoutubeItem('${Utils.escapeAttr(item.url)}')">${icon('play', 14)} Play</button>
                <button class="btn-panel btn-stop" onclick="pauseYoutubeItem('${Utils.escapeAttr(item.url)}')">${icon('pause', 14)} Pause</button>
            </div>
            
            <div class="panel-actions">
                 <button class="btn-panel btn-stop" onclick="stopYoutube()">${icon('stop', 14)} Stop Playback</button>
            </div>
            
            <div class="panel-actions">
                <button class="btn-panel btn-stop" style="color: var(--primary); border-color: var(--primary);" onclick="saveYoutubeAsSound('${Utils.escapeAttr(item.url)}')">${icon('add', 14)} Save as Sound</button>
            </div>

            <div class="panel-actions">
                <button class="btn-panel btn-delete" onclick="deleteYoutubeItem('${Utils.escapeAttr(item.url)}')">${icon('trash', 14)} Delete</button>
            </div>
        `;

        // Initialize YouTube waveform visualizer after DOM is ready
        setTimeout(() => {
            if (window.YouTubeRangeSlider) {
                new YouTubeRangeSlider('youtube-waveform-container', item.url);
            }
            // Setup effects event listeners
            EffectsEvents.setupEffectsListeners('youtube', item.url, AppState.getYoutubeEffects(item.url));
        }, 100);
    },

    /**
     * Render TikTok panel
     * @param {Object} item - TikTok item {url, title}
     */
    renderTikTokPanel(item) {
        const panel = document.getElementById('right-panel');
        const displayName = AppState.getTikTokDisplayName(item.url, item.title);

        panel.innerHTML = `
            ${UIHelpers.buildPanelHeader({
            id: 'tt-name-input',
            displayName: displayName,
            placeholder: item.title,
            onChangeHandler: 'TikTokEvents.onNameChange',
            infoText: item.url
        })}
            
            ${UIHelpers.buildPreviewWave(true)}
            
            ${UIHelpers.buildKeybindSection({
            inputId: 'tt-keybind-input',
            value: AppState.getTikTokKeybind(item.url),
            onClickHandler: 'TikTokEvents.startKeybindRecording'
        })}
            
            ${UIHelpers.buildVolumeSection({
            sliderId: 'tiktok-volume',
            valueId: 'tiktok-volume-value',
            volume: AppState.getTikTokVolume(item.url),
            title: 'TikTok Volume',
            onInputHandler: 'TikTokEvents.onVolumeLive',
            onChangeHandler: 'TikTokEvents.onVolumeSave'
        })}
            
            ${UIHelpers.buildTrimSection('tiktok-waveform-container')}

            ${UIHelpers.buildScreamToggle({
            checkboxId: 'tt-scream-checkbox',
            isActive: AppState.isTikTokScreamMode(item.url),
            onChangeHandler: 'TikTokEvents.toggleScreamMode',
            url: item.url
        })}
            
            ${UIHelpers.buildChipmunkToggle({
            checkboxId: 'tt-pitch-checkbox',
            isActive: AppState.isTikTokPitchMode(item.url),
            onChangeHandler: 'TikTokEvents.togglePitchMode',
            url: item.url
        })}

            ${UIHelpers.buildLoopToggle({
            checkboxId: 'tt-loop-checkbox',
            isActive: AppState.isTikTokLoop(item.url),
            onChangeHandler: 'TikTokEvents.toggleLoop',
            url: item.url
        })}
            
            ${EffectsHelpers.renderEffectsSection(AppState.getTikTokEffects(item.url))}
            
            <div class="panel-actions">
                <button class="btn-panel btn-play" onclick="TikTokEvents.playItem('${Utils.escapeAttr(item.url)}')">${icon('play', 14)} Play</button>
                <button class="btn-panel btn-stop" onclick="TikTokEvents.pauseItem('${Utils.escapeAttr(item.url)}')">${icon('pause', 14)} Pause</button>
            </div>
            
            <div class="panel-actions">
                 <button class="btn-panel btn-stop" onclick="TikTokEvents.stop()">${icon('stop', 14)} Stop Playback</button>
            </div>
            
             <div class="panel-actions">
                <button class="btn-panel btn-stop" style="color: var(--primary); border-color: var(--primary);" onclick="TikTokEvents.saveAsSound()">${icon('add', 14)} Save as Sound</button>
            </div>

            <div class="panel-actions">
                <button class="btn-panel btn-delete" onclick="TikTokEvents.deleteItem('${Utils.escapeAttr(item.url)}')">${icon('trash', 14)} Delete</button>
            </div>
        `;

        // Initialize TikTok waveform visualizer after DOM is ready
        setTimeout(() => {
            if (window.TikTokRangeSlider) {
                new TikTokRangeSlider('tiktok-waveform-container', item.url);
            }
            // Setup effects event listeners
            EffectsEvents.setupEffectsListeners('tiktok', item.url, AppState.getTikTokEffects(item.url));
        }, 100);
    },

    /**
     * Show empty panel placeholder
     */
    showEmptyPanel() {
        document.getElementById('right-panel').innerHTML = `
            <div class="panel-placeholder">
                <span class="placeholder-icon">${IconManager.get('waveform', { size: 64 })}</span>
                <p>Click a sound to edit</p>
            </div>
        `;
    }
};

window.PanelRenderer = PanelRenderer;
