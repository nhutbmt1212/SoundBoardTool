// UI Rendering for Soundboard Pro

const UI = {
    // Initialize icons
    initIcons() {
        const titleIcon = document.getElementById('title-icon');
        if (titleIcon) titleIcon.innerHTML = Icons.music;

        const stopIcon = document.getElementById('stop-icon');
        if (stopIcon) stopIcon.innerHTML = Icons.stop;

        const placeholder = document.getElementById('placeholder-icon');
        if (placeholder) placeholder.innerHTML = Icons.waveform;

        console.log('[DEBUG] Icons initialized');
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
                    <span class="empty-icon">${Icons.folder}</span>
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
        if (isScream) badges.push('😈');
        if (isPitch) badges.push('🐿️');

        return `
            <div class="sound-card ${isScream ? 'scream-mode' : ''} ${isPitch ? 'pitch-mode' : ''}" data-name="${Utils.escapeAttr(name)}">
                <div class="sound-thumbnail">
                    <span class="thumb-icon">${Icons.waveform}</span>
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
                <div class="panel-section-title">😈 Scream Mode</div>
                <label class="scream-toggle">
                    <input type="checkbox" id="scream-checkbox" ${isScream ? 'checked' : ''} onchange="EventHandlers.toggleScreamMode()">
                    <span class="scream-slider"></span>
                    <span class="scream-label">${isScream ? 'ON - 5000% BOOST! 💀' : 'OFF'}</span>
                </label>
                <div class="scream-hint">Boost volume to max for trolling</div>
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">🐿️ Chipmunk Mode</div>
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
                <span class="placeholder-icon">${Icons.waveform}</span>
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
            text.textContent = AppState.micEnabled ? 'Mic ON' : 'Mic OFF';
        }
    },

    // Update YouTube UI
    updateYoutubeUI(playing, title = '') {
        const infoEl = document.getElementById('youtube-info');
        if (playing && title) {
            infoEl.textContent = '▶ ' + title;
            infoEl.className = 'youtube-info playing';
        } else {
            infoEl.textContent = '';
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
        infoEl.textContent = '❌ ' + error;
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
                <div class="sound-thumbnail youtube-thumbnail" style="width: 120px; height: 80px;">
                    <span class="thumb-icon" style="opacity: 1;">📺</span>
                </div>
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
                    <span class="empty-icon">📺</span>
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
        const keybind = item.keybind || '';
        const isCurrentUrl = info && info.url === item.url;
        const isPlaying = isCurrentUrl && info.playing;
        const isPaused = isCurrentUrl && info.paused;

        // Determine button state
        let actionBtn;
        if (isPlaying && !isPaused) {
            // Playing -> Show Pause
            actionBtn = `
                <button class="btn-yt-action pause" onclick="event.stopPropagation(); pauseYoutubeItem('${Utils.escapeAttr(item.url)}')" title="Pause">
                    ${icon('pause', 12)}
                </button>
            `;
        } else {
            // Paused or Stopped -> Show Play
            // If Paused, Play will Resume
            actionBtn = `
                <button class="btn-yt-action play" onclick="event.stopPropagation(); playYoutubeItem('${Utils.escapeAttr(item.url)}')" title="${isPaused ? 'Resume' : 'Play'}">
                    ${icon('play', 12)}
                </button>
            `;
        }

        return `
            <div class="sound-card youtube-item ${isPlaying ? 'playing' : ''} ${isPaused ? 'paused' : ''}" data-url="${item.url}">
                <div class="sound-thumbnail youtube-thumbnail">
                    <span class="thumb-icon">📺</span>
                    ${isPlaying && !isPaused ? '<div class="playing-indicator">▶</div>' : ''}
                    ${isPaused ? '<div class="playing-indicator">⏸</div>' : ''}
                </div>
                <div class="sound-name" title="${Utils.escapeAttr(item.title)}">${Utils.escapeHtml(item.title)}</div>
                
                <div class="youtube-actions">
                    ${actionBtn}
                    <button class="btn-yt-action delete" onclick="event.stopPropagation(); deleteYoutubeItem('${Utils.escapeAttr(item.url)}')" title="Delete">
                        ${icon('trash', 12)}
                    </button>
                </div>
            </div>
        `;
    }
};

window.UI = UI;
