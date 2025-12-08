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
        setIcon('icon-tab-sounds', 'music', 16);
        setIcon('icon-tab-youtube', 'youtube', 16);
        setIcon('icon-tab-tiktok', 'tiktok', 16);

        // Actions
        setIcon('icon-btn-add', 'add', 14);
        setIcon('icon-btn-refresh', 'refresh', 14);
        setIcon('icon-btn-add-yt', 'add', 14);
        setIcon('icon-btn-add-tt', 'add', 14);
        setIcon('icon-btn-refresh-yt', 'refresh', 14);
        setIcon('icon-btn-refresh-tt', 'refresh', 14);

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

        // Sort sounds alphabetically by display name
        const sortedSounds = [...sounds].sort((a, b) => {
            const nameA = AppState.getDisplayName(a).toLowerCase();
            const nameB = AppState.getDisplayName(b).toLowerCase();
            return nameA.localeCompare(nameB);
        });

        grid.innerHTML = sortedSounds.map(name => this.renderSoundCard(name)).join('');
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
                <div class="panel-section-title">${icon('clock', 16)} Trim Audio</div>
                <div id="waveform-container"></div>
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

        // Initialize waveform visualizer after DOM is ready
        setTimeout(() => {
            if (window.RangeSlider) {
                new RangeSlider('waveform-container', name);
            }
        }, 100);
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

    // Unified Now Playing Bar
    currentPlayingType: null,

    updateNowPlaying(title, type, isPlaying, isPaused = false) {
        const iconContainer = document.getElementById('now-playing-icon');
        const titleEl = document.getElementById('now-playing-title');
        const statusIcon = document.getElementById('now-playing-status-icon');
        const progressFill = document.getElementById('now-playing-progress');

        if (!title && !isPlaying) {
            // Only clear if the current type matches the request type
            // This prevents 'youtube' stop event from clearing 'sound' play event logic if race condition
            if (title === null && (this.currentPlayingType === type || this.currentPlayingType === null)) {
                titleEl.textContent = 'Select a sound...';
                titleEl.classList.remove('active');
                iconContainer.classList.remove('playing');
                iconContainer.innerHTML = IconManager.get('waveform', { size: 24 });
                statusIcon.innerHTML = '';
                progressFill.style.width = '0%';
                this.currentPlayingType = null;
            }
            return;
        }

        // Active playing or paused state
        this.currentPlayingType = type;

        // Update Title
        titleEl.textContent = title || 'Unknown Track';
        titleEl.classList.add('active');

        // Update Icon based on type
        let typeIcon = 'waveform';
        if (type === 'youtube') typeIcon = 'youtube';
        if (type === 'tiktok') typeIcon = 'tiktok';

        iconContainer.innerHTML = IconManager.get(typeIcon, { size: 24 });

        // Update Playing Status
        const btnHtml = `
            <button class="btn-control-main" id="btn-now-playing-toggle" onclick="EventHandlers.togglePlayback()">
                ${IconManager.get(isPlaying && !isPaused ? 'pause' : 'play', { size: 16 })}
            </button>
        `;

        // Only update if changed to prevent flicker/event loss (though onclick is inline)
        if (statusIcon.innerHTML !== btnHtml) {
            statusIcon.innerHTML = btnHtml;
        }

        if (isPlaying && !isPaused) {
            iconContainer.classList.add('playing');
            progressFill.style.width = '100%';
        } else {
            iconContainer.classList.remove('playing');
            progressFill.style.width = '0%';
        }
    },

    // Update playing state on cards (Local Sounds)
    updatePlayingState(playingSound) {
        document.querySelectorAll('.sound-card').forEach(card => {
            // Only add playing class if playingSound matches, otherwise remove
            if (playingSound && card.dataset.name === playingSound) {
                card.classList.add('playing');
            } else {
                card.classList.remove('playing');
            }
        });

        if (playingSound) {
            const displayName = AppState.getDisplayName(playingSound);
            this.updateNowPlaying(displayName, 'sound', true);
        } else {
            this.updateNowPlaying(null, 'sound', false);
        }
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
    updateYoutubeUI(playing, title = '', paused = false) {
        if (playing && title) {
            // Always update now-playing-bar when playing
            this.updateNowPlaying(title, 'youtube', true, paused);
        } else if (!playing) {
            // Clear now-playing-bar when stopped
            this.updateNowPlaying(null, 'youtube', false);
        }
    },

    // Set YouTube loading state
    setYoutubeLoading() {
        this.updateNowPlaying('Loading YouTube...', 'youtube', false);
        const playBtn = document.getElementById('btn-youtube-play');
        if (playBtn) playBtn.disabled = true;
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
        const displayName = AppState.getYoutubeDisplayName(item.url, item.title);

        panel.innerHTML = `
            <div class="panel-header">
                <input type="text" class="panel-sound-name editable" id="yt-name-input" 
                       value="${Utils.escapeAttr(displayName)}" 
                       placeholder="${Utils.escapeAttr(item.title)}"
                       onchange="EventHandlers.onYoutubeNameChange(this.value)"
                       onkeydown="if(event.key==='Enter')this.blur()">
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
                <div class="panel-section-title">${icon('clock', 16)} Trim Audio</div>
                <div id="youtube-waveform-container"></div>
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
        }, 100);
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

        // Sort YouTube items alphabetically by display name
        const sortedItems = [...items].sort((a, b) => {
            const nameA = AppState.getYoutubeDisplayName(a.url, a.title).toLowerCase();
            const nameB = AppState.getYoutubeDisplayName(b.url, b.title).toLowerCase();
            return nameA.localeCompare(nameB);
        });

        grid.innerHTML = sortedItems.map(item => this.renderYoutubeItem(item, info)).join('');
    },

    // Render single YouTube item
    // Add loading YouTube card
    addLoadingYoutubeCard(url) {
        const grid = document.getElementById('youtube-grid');
        if (!grid) return;

        // Remove empty state
        const empty = grid.querySelector('.empty-state');
        if (empty) empty.remove();

        const card = document.createElement('div');
        card.className = 'sound-card youtube-item loading';
        card.dataset.url = 'loading-' + url;
        card.innerHTML = `
            <div class="sound-thumbnail youtube-thumbnail">
                <div class="loading-pie" style="--p: 0;"></div>
                <div class="loading-percent">0%</div>
            </div>
            <div class="sound-name">Downloading...</div>
            <div class="sound-info" style="font-size: 10px; color: var(--text-muted); padding: 0 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${Utils.escapeHtml(url)}</div>
        `;

        grid.prepend(card);
    },

    // Update loading progress
    updateYoutubeProgress(url, percent) {
        const grid = document.getElementById('youtube-grid');
        if (!grid) return;

        const card = grid.querySelector(`.youtube-item[data-url="loading-${Utils.escapeAttr(url)}"]`);
        if (!card) return;

        const pie = card.querySelector('.loading-pie');
        const text = card.querySelector('.loading-percent');

        if (pie) pie.style.setProperty('--p', percent);
        if (text) text.textContent = Math.round(percent) + '%';
    },

    // Remove loading YouTube card
    removeLoadingYoutubeCard(url) {
        const grid = document.getElementById('youtube-grid');
        if (!grid) return;
        const card = grid.querySelector(`.youtube-item[data-url="loading-${Utils.escapeAttr(url)}"]`);
        if (card) card.remove();
    },

    renderYoutubeItem(item, info) {
        // Use AppState as source of truth for keybinds and display name
        const keybind = AppState.getYoutubeKeybind(item.url) || item.keybind || '';
        const displayName = AppState.getYoutubeDisplayName(item.url, item.title);
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
                <div class="sound-name" title="${Utils.escapeAttr(displayName)}">${Utils.escapeHtml(displayName)}</div>
                
                <div class="sound-keybind ${keybind ? 'has-bind' : ''}">
                    ${keybind || 'Add keybind'}
                </div>
            </div>
        `;
    },

    // Update TikTok UI
    updateTikTokUI(playing, title = '', paused = false) {
        if (playing && title) {
            // Always update now-playing-bar when playing
            this.updateNowPlaying(title, 'tiktok', true, paused);
        } else if (!playing) {
            // Clear now-playing-bar when stopped
            this.updateNowPlaying(null, 'tiktok', false);
        }
    },

    // Set TikTok loading state
    setTikTokLoading() {
        this.updateNowPlaying('Loading TikTok...', 'tiktok', false);
    },

    // Set TikTok error state
    setTikTokError(error) {
        const infoEl = document.getElementById('tiktok-info');
        infoEl.innerHTML = IconManager.get('warning', { size: 14 }) + ' ' + Utils.escapeHtml(error);
        infoEl.className = 'tiktok-info error';
    },

    // Enable TikTok play button
    enableTikTokPlayBtn() {
    },

    // Select TikTok card visually
    selectTikTokCard(url) {
        document.querySelectorAll('.tiktok-item').forEach(card => {
            card.classList.toggle('selected', card.dataset.url === url);
        });
    },

    // Show panel for TikTok item
    showTikTokPanel(item) {
        const panel = document.getElementById('right-panel');
        const displayName = AppState.getTikTokDisplayName(item.url, item.title);

        panel.innerHTML = `
            <div class="panel-header">
                <input type="text" class="panel-sound-name editable" id="tt-name-input" 
                       value="${Utils.escapeAttr(displayName)}" 
                       placeholder="${Utils.escapeAttr(item.title)}"
                       onchange="TikTokEvents.onNameChange(this.value)"
                       onkeydown="if(event.key==='Enter')this.blur()">
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
                <input type="text" class="keybind-input" id="tt-keybind-input" 
                       value="${AppState.getTikTokKeybind(item.url)}" 
                       placeholder="Click to set keybind"
                       readonly
                       onclick="TikTokEvents.startKeybindRecording()">
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${icon('volume', 16)} TikTok Volume</div>
                <div class="volume-control">
                    <input type="range" class="volume-slider" id="tiktok-volume" 
                           min="0" max="100" value="100"
                           oninput="TikTokEvents.onVolumeChange(this.value)">
                    <span class="volume-value" id="tiktok-volume-value">100%</span>
                </div>
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${icon('clock', 16)} Trim Audio</div>
                <div id="tiktok-waveform-container"></div>
            </div>

            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('scream', { size: 16 })} Scream Mode</div>
                <label class="scream-toggle">
                    <input type="checkbox" id="tt-scream-checkbox" ${AppState.isTikTokScreamMode(item.url) ? 'checked' : ''} onchange="TikTokEvents.toggleScreamMode('${Utils.escapeAttr(item.url)}')">
                    <span class="scream-slider"></span>
                    <span class="scream-label">${AppState.isTikTokScreamMode(item.url) ? 'ON - 5000% BOOST!' : 'OFF'}</span>
                </label>
                <div class="scream-hint">Boost volume to max for trolling</div>
            </div>
            
            <div class="panel-section">
                <div class="panel-section-title">${IconManager.get('chipmunk', { size: 16 })} Chipmunk Mode</div>
                <label class="pitch-toggle">
                    <input type="checkbox" id="tt-pitch-checkbox" ${AppState.isTikTokPitchMode(item.url) ? 'checked' : ''} onchange="TikTokEvents.togglePitchMode('${Utils.escapeAttr(item.url)}')">
                    <span class="pitch-slider"></span>
                    <span class="pitch-label">${AppState.isTikTokPitchMode(item.url) ? 'ON - HIGH PITCH!' : 'OFF'}</span>
                </label>
                <div class="pitch-hint">Speed up audio for chipmunk voice</div>
            </div>
            
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
        }, 100);
    },

    // Render TikTok grid
    renderTikTokGrid(items, info) {
        const grid = document.getElementById('tiktok-grid');

        if (!items || items.length === 0) {
            grid.innerHTML = `
                <div class="empty-state">
                    <span class="empty-icon">${IconManager.get('tiktok', { size: 64 })}</span>
                    <h2>No TikTok videos</h2>
                    <p>Click "Add from TikTok" to get started</p>
                </div>
            `;
            return;
        }

        // Sort TikTok items alphabetically by display name
        const sortedItems = [...items].sort((a, b) => {
            const nameA = AppState.getTikTokDisplayName(a.url, a.title).toLowerCase();
            const nameB = AppState.getTikTokDisplayName(b.url, b.title).toLowerCase();
            return nameA.localeCompare(nameB);
        });

        grid.innerHTML = sortedItems.map(item => this.renderTikTokItem(item, info)).join('');
    },

    // Render single TikTok item
    renderTikTokItem(item, info) {
        const keybind = AppState.getTikTokKeybind(item.url) || item.keybind || '';
        const displayName = AppState.getTikTokDisplayName(item.url, item.title);
        const isCurrentUrl = info && info.url === item.url;
        const isPlaying = isCurrentUrl && info.playing;
        const isPaused = isCurrentUrl && info.paused;

        return `
            <div class="sound-card tiktok-item ${isPlaying ? 'playing' : ''} ${isPaused ? 'paused' : ''}" data-url="${Utils.escapeAttr(item.url)}">
                <div class="sound-thumbnail youtube-thumbnail">
                    <span class="thumb-icon">${IconManager.get('waveform')}</span>
                    ${isPlaying && !isPaused ? `<div class="playing-indicator">${IconManager.get('playCircle', { size: 32 })}</div>` : ''}
                    ${isPaused ? `<div class="playing-indicator">${IconManager.get('pauseCircle', { size: 32 })}</div>` : ''}
                </div>
                <div class="sound-name" title="${Utils.escapeAttr(displayName)}">${Utils.escapeHtml(displayName)}</div>
                
                <div class="sound-keybind ${keybind ? 'has-bind' : ''}">
                    ${keybind || 'Add keybind'}
                </div>
            </div>
        `;
    },

    // Add loading TikTok card
    addLoadingTikTokCard(url) {
        const grid = document.getElementById('tiktok-grid');
        if (!grid) return;

        // Remove empty state
        const empty = grid.querySelector('.empty-state');
        if (empty) empty.remove();

        const card = document.createElement('div');
        card.className = 'sound-card tiktok-item loading';
        card.dataset.url = 'loading-' + url;
        card.innerHTML = `
            <div class="sound-thumbnail youtube-thumbnail">
                <div class="loading-pie" style="--p: 0;"></div>
                <div class="loading-percent">0%</div>
            </div>
            <div class="sound-name">Downloading...</div>
            <div class="sound-info" style="font-size: 10px; color: var(--text-muted); padding: 0 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${Utils.escapeHtml(url)}</div>
        `;

        grid.prepend(card);
    },

    // Update loading progress
    updateTikTokProgress(url, percent) {
        const grid = document.getElementById('tiktok-grid');
        if (!grid) return;

        const card = grid.querySelector(`.tiktok-item[data-url="loading-${Utils.escapeAttr(url)}"]`);
        if (!card) return;

        const pie = card.querySelector('.loading-pie');
        const text = card.querySelector('.loading-percent');

        if (pie) pie.style.setProperty('--p', percent);
        if (text) text.textContent = Math.round(percent) + '%';
    },

    // Remove loading TikTok card
    removeLoadingTikTokCard(url) {
        const grid = document.getElementById('tiktok-grid');
        if (!grid) return;
        const card = grid.querySelector(`.tiktok-item[data-url="loading-${Utils.escapeAttr(url)}"]`);
        if (card) card.remove();
    },

    // Modal
    closeModalCallback: null,

    showModal({ title, body, onConfirm, confirmText = 'Confirm', showCancel = true }) {
        const overlay = document.getElementById('modal-overlay');
        const titleEl = document.getElementById('modal-title');
        const bodyEl = document.getElementById('modal-body');
        const confirmBtn = document.getElementById('modal-confirm-btn');
        const footer = document.getElementById('modal-footer');

        if (!overlay) return;

        titleEl.textContent = title;
        bodyEl.innerHTML = body;
        confirmBtn.textContent = confirmText;

        // Reset footer
        footer.style.display = 'flex';

        // Hide cancel button if needed? Using CSS might be better but let's leave it for now
        // For simple prompts we want Cancel.

        // Clean previous listeners
        const newConfirmBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);

        newConfirmBtn.onclick = () => {
            if (onConfirm) onConfirm();
            this.closeModal();
        };

        // Handle Enter key in inputs
        const inputs = bodyEl.querySelectorAll('input');
        if (inputs.length > 0) {
            inputs[0].focus();
            inputs[0].addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    newConfirmBtn.click();
                }
            });
            // Focus first input after slight delay for animation
            setTimeout(() => inputs[0].focus(), 50);
        }

        overlay.classList.add('active');
        this.closeModalCallback = null; // Reset callback
    },

    closeModal() {
        const overlay = document.getElementById('modal-overlay');
        if (overlay) overlay.classList.remove('active');
        if (this.closeModalCallback) this.closeModalCallback();
    }
};

window.UI = UI;
