// UI Rendering for Soundboard Pro

const UI = {
    // Initialize icons
    initIcons() {
        const setIcon = (id, name, size = 20) => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = IconManager.get(name, { size });
        };

        setIcon('title-icon', 'music', 28);
        setIcon('stop-icon', 'stop', 20);
        setIcon('placeholder-icon', 'waveform', 64);

        // Tabs
        setIcon('icon-tab-sounds', 'music', 16);
        setIcon('icon-tab-youtube', 'youtube', 16);

        // Actions
        setIcon('icon-btn-add', 'add', 14);
        setIcon('icon-btn-refresh', 'refresh', 14);
        setIcon('icon-btn-add-yt', 'youtube', 14); // Or add
        setIcon('icon-btn-refresh-yt', 'refresh', 14);

        // Mic
        setIcon('mic-icon', 'micOff', 16); // Default off
    },

    // Update Stop All keybind UI
    updateStopKeybindUI() {
        const el = document.getElementById('stop-keybind');
        const textEl = document.getElementById('stop-keybind-text');
        if (el && textEl) {
            if (AppState.stopAllKeybind) {
                el.classList.add('has-bind');
                textEl.textContent = AppState.stopAllKeybind;
            } else {
                el.classList.remove('has-bind');
                textEl.textContent = 'Add keybind';
            }
        }
    },

    // Render sound grid
    renderSoundGrid(sounds) {
        const grid = document.getElementById('sounds-grid');

        if (sounds.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">${IconManager.get('folder', { size: 64 })}</span>
                    <h2>No sounds found</h2>
                    <p>Click ${icon('add', 16)} to add sounds</p>
                </div>
            `;
            return;
        }

        grid.innerHTML = sounds.map(name => this.renderSoundCard(name)).join('');
    },

    // Render single sound card
    renderSoundCard(name) {
        const keybind = AppState.getKeybind(name);
        const isScream = AppState.isScreamMode(name);
        const isPitch = AppState.isPitchMode(name);
        const displayName = AppState.getDisplayName(name);

        const badges = [];
        if (isScream) badges.push(IconManager.get('scream', { size: 14 }));
        if (isPitch) badges.push(IconManager.get('chipmunk', { size: 14 }));

        return `
            <div class="sound-card ${isScream ? 'scream-mode' : ''} ${isPitch ? 'pitch-mode' : ''}" data-name="${Utils.escapeAttr(name)}">
                <div class="sound-thumbnail">
                    <span class="thumb-icon">${IconManager.get('waveform')}</span>
                    ${badges.length ? `<span class="mode-badges">${badges.join('')}</span>` : ''}
                </div>
                <div class="sound-name" title="${Utils.escapeAttr(displayName)}">${Utils.escapeHtml(displayName)}</div>
                <div class="sound-keybind ${keybind ? 'has-bind' : ''}">
                    ${keybind || 'Add keybind'}
                </div>
            </div>
        `;
    },

    // Select sound card visually
    selectSoundCard(name) {
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.toggle('selected', card.dataset.name === name);
        });
    },

    // Show right panel for selected sound
    showSoundPanel(name) {
        const panel = document.getElementById('right-panel');
        const volume = AppState.getVolume(name);
        const keybind = AppState.getKeybind(name);
        const isScream = AppState.isScreamMode(name);
        const isPitch = AppState.isPitchMode(name);
        const displayName = AppState.getDisplayName(name);

        panel.innerHTML = `
            <div class="panel-header">
                <input type="text" class="panel-sound-name editable" id="sound-name-input" 
                       value="${Utils.escapeAttr(displayName)}" 
                       placeholder="${Utils.escapeAttr(name)}"
                       onchange="EventHandlers.onSoundNameChange(this.value)"
                       onkeydown="if(event.key==='Enter')this.blur()">
                <div class="panel-sound-info">${Utils.escapeHtml(name)}</div>
            </div>
            
            <div class="panel-preview">
                <div class="preview-wave ${isScream ? 'scream-active' : ''}">
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
            
            <div class="panel-section">
                <div class="panel-section-title">${icon('keyboard', 16)} Keybind</div>
                <input type="text" class="keybind-input" id="keybind-input" 
                       value="${keybind}" 
                       placeholder="Click to set keybind"
                       readonly
                       onclick="EventHandlers.startKeybindRecordPanel()">
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${icon('volume', 16)} Volume</div>
                <div class="volume-control">
                    <input type="range" class="volume-slider" id="sound-volume" 
                           min="0" max="100" value="${volume}"
                           oninput="EventHandlers.onSoundVolumeChange(this.value)">
                    <span class="volume-value" id="volume-value">${volume}</span>
                </div>
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('scream', { size: 16 })} Scream Mode</div>
                <label class="scream-toggle">
                    <input type="checkbox" id="scream-checkbox" ${isScream ? 'checked' : ''} onchange="EventHandlers.toggleScreamMode()">
                    <span class="scream-slider"></span>
                    <span class="scream-label">${isScream ? 'ON - 5000% BOOST!' : 'OFF'}</span>
                </label>
                <div class="scream-hint">Boost volume to max for trolling</div>
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('chipmunk', { size: 16 })} Chipmunk Mode</div>
                <label class="pitch-toggle">
                    <input type="checkbox" id="pitch-checkbox" ${isPitch ? 'checked' : ''} onchange="EventHandlers.togglePitchMode()">
                    <span class="pitch-slider"></span>
                    <span class="pitch-label">${isPitch ? 'ON - HIGH PITCH!' : 'OFF'}</span>
                </label>
                <div class="pitch-hint">Speed up audio for chipmunk voice</div>
            </div>
            
            <div class="panel-actions">
                <button class="btn-panel btn-play" onclick="EventHandlers.playSound('${Utils.escapeAttr(name)}')">${icon('play', 14)} Play</button>
                <button class="btn-panel btn-stop" onclick="EventHandlers.stopAll()">${icon('stop', 14)} Stop</button>
            </div>
            
            <div class="panel-actions">
                <button class="btn-panel btn-delete" onclick="EventHandlers.deleteSound('${Utils.escapeAttr(name)}')">${icon('trash', 14)} Delete</button>
            </div>
        `;
    },

    // Show empty panel
    showEmptyPanel() {
        document.getElementById('right-panel').innerHTML = `
            <div class="panel-placeholder">
                <span class="placeholder-icon">${IconManager.get('waveform', { size: 64 })}</span>
                <p>Click a sound to edit</p>
            </div>
        `;
    },

    // Update playing state on cards
    updatePlayingState(playingSound) {
        document.querySelectorAll('.sound-card').forEach(card => {
            // Only add playing class if playingSound matches, otherwise remove
            if (playingSound && card.dataset.name === playingSound) {
                card.classList.add('playing');
            } else {
                card.classList.remove('playing');
            }
        });
    },

    // Clear all playing states
    clearPlayingStates() {
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.remove('playing');
        });
    },

    // Update mic UI
    updateMicUI() {
        const btn = document.getElementById('btn-mic');
        const text = document.getElementById('mic-text');
        if (btn && text) {
            btn.classList.toggle('active', AppState.micEnabled);
            const iconEl = document.getElementById('mic-icon');
            if (iconEl) iconEl.innerHTML = IconManager.get(AppState.micEnabled ? 'mic' : 'micOff', { size: 16 });
            text.textContent = AppState.micEnabled ? 'Mic ON' : 'Mic OFF';
        }
    },

    // Update YouTube UI
    updateYoutubeUI(playing, title = '') {
        const infoEl = document.getElementById('youtube-info');
        if (playing && title) {
            infoEl.innerHTML = IconManager.get('play', { size: 14 }) + ' ' + Utils.escapeHtml(title);
            infoEl.className = 'youtube-info playing';
        } else {
            infoEl.innerHTML = '';
            infoEl.className = 'youtube-info';
        }
    },

    // Set YouTube loading state
    setYoutubeLoading() {
        const infoEl = document.getElementById('youtube-info');
        const playBtn = document.getElementById('btn-youtube-play');
        infoEl.textContent = 'Loading...';
        infoEl.className = 'youtube-info loading';
        playBtn.disabled = true;
    },

    // Set YouTube error state
    setYoutubeError(error) {
        const infoEl = document.getElementById('youtube-info');
        infoEl.innerHTML = IconManager.get('warning', { size: 14 }) + ' ' + Utils.escapeHtml(error);
        infoEl.className = 'youtube-info error';
    },

    // Enable YouTube play button
    enableYoutubePlayBtn() {
        document.getElementById('btn-youtube-play').disabled = false;
    },

    // Select YouTube card visually
    selectYoutubeCard(url) {
        document.querySelectorAll('.youtube-item').forEach(card => {
            card.classList.toggle('selected', card.dataset.url === url);
        });
    },

    // Show panel for YouTube item
    showYoutubePanel(item) {
        const panel = document.getElementById('right-panel');

        panel.innerHTML = `
            <div class="panel-header">
                <div class="panel-sound-name" title="${Utils.escapeAttr(item.title)}">${Utils.escapeHtml(item.title)}</div>
                <div class="panel-sound-info" title="${Utils.escapeAttr(item.url)}">${Utils.escapeHtml(item.url)}</div>
            </div>
            
            <div class="panel-preview">
                <div class="preview-wave scream-active">
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
            
            <div class="panel-section">
                <div class="panel-section-title">${icon('keyboard', 16)} Keybind</div>
                <input type="text" class="keybind-input" id="yt-keybind-input" 
                       value="${AppState.getYoutubeKeybind(item.url)}" 
                       placeholder="Click to set keybind"
                       readonly
                       onclick="EventHandlers.startYoutubeKeybindRecording()">
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${icon('volume', 16)} Global YouTube Volume</div>
                <div class="volume-control">
                    <input type="range" class="volume-slider" id="youtube-volume" 
                           min="0" max="100" value="100"
                           oninput="onYoutubeVolumeChange(this.value)">
                    <span class="volume-value" id="youtube-volume-value">100%</span>
                </div>
            </div>

            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('scream', { size: 16 })} Scream Mode</div>
                <label class="scream-toggle">
                    <input type="checkbox" id="yt-scream-checkbox" ${AppState.isYoutubeScreamMode(item.url) ? 'checked' : ''} onchange="EventHandlers.toggleYoutubeScreamMode('${Utils.escapeAttr(item.url)}')">
                    <span class="scream-slider"></span>
                    <span class="scream-label">${AppState.isYoutubeScreamMode(item.url) ? 'ON - 5000% BOOST!' : 'OFF'}</span>
                </label>
                <div class="scream-hint">Boost volume to max for trolling</div>
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('chipmunk', { size: 16 })} Chipmunk Mode</div>
                <label class="pitch-toggle">
                    <input type="checkbox" id="yt-pitch-checkbox" ${AppState.isYoutubePitchMode(item.url) ? 'checked' : ''} onchange="EventHandlers.toggleYoutubePitchMode('${Utils.escapeAttr(item.url)}')">
                    <span class="pitch-slider"></span>
                    <span class="pitch-label">${AppState.isYoutubePitchMode(item.url) ? 'ON - HIGH PITCH!' : 'OFF'}</span>
                </label>
                <div class="pitch-hint">Speed up audio for chipmunk voice</div>
            </div>
            
            <div class="panel-actions">
                <button class="btn-panel btn-play" onclick="playYoutubeItem('${Utils.escapeAttr(item.url)}')">${icon('play', 14)} Play</button>
                <button class="btn-panel btn-stop" onclick="pauseYoutubeItem('${Utils.escapeAttr(item.url)}')">${icon('pause', 14)} Pause</button>
            </div>
            
            <div class="panel-actions">
                 <button class="btn-panel btn-stop" onclick="stopYoutube()">${icon('stop', 14)} Stop Playback</button>
            </div>
            
            <div class="panel-actions">
                <button class="btn-panel btn-stop" style="color: var(--primary); border-color: var(--primary);" onclick="window.saveYoutubeAsSound()">${icon('add', 14)} Save as Sound</button>
            </div>

            <div class="panel-actions">
                <button class="btn-panel btn-delete" onclick="deleteYoutubeItem('${Utils.escapeAttr(item.url)}')">${icon('trash', 14)} Delete</button>
            </div>
        `;
    },

    // Render YouTube grid
    renderYoutubeGrid(items, info) {
        const grid = document.getElementById('youtube-grid');

        if (!items || items.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">${IconManager.get('youtube', { size: 64 })}</span>
                    <h2>No YouTube videos</h2>
                    <p>Click "Add from YouTube" to get started</p>
                </div>
            `;
            return;
        }

        grid.innerHTML = items.map(item => this.renderYoutubeItem(item, info)).join('');
    },

    // Render single YouTube item
    renderYoutubeItem(item, info) {
        // Use AppState as source of truth for keybinds
        const keybind = AppState.getYoutubeKeybind(item.url) || item.keybind || '';
        const isCurrentUrl = info && info.url === item.url;
        const isPlaying = isCurrentUrl && info.playing;
        const isPaused = isCurrentUrl && info.paused;

        return `
            <div class="sound-card youtube-item ${isPlaying ? 'playing' : ''} ${isPaused ? 'paused' : ''}" data-url="${Utils.escapeAttr(item.url)}">
                <div class="sound-thumbnail youtube-thumbnail">
                    <span class="thumb-icon">${IconManager.get('waveform')}</span>
                    ${isPlaying && !isPaused ? `<div class="playing-indicator">${IconManager.get('playCircle', { size: 32 })}</div>` : ''}
                    ${isPaused ? `<div class="playing-indicator">${IconManager.get('pauseCircle', { size: 32 })}</div>` : ''}
                </div>
                <div class="sound-name" title="${Utils.escapeAttr(item.title)}">${Utils.escapeHtml(item.title)}</div>
                
                <div class="sound-keybind ${keybind ? 'has-bind' : ''}">
                    ${keybind || 'Add keybind'}
                </div>
            </div>
        `;
    }
};

window.UI = UI;
