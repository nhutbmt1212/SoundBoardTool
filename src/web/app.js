// Soundboard Pro - Frontend Logic

let selectedSound = null;
let soundVolumes = {};
let soundKeybinds = {};
let soundScreamMode = {};  // Scream mode per sound
let soundPitchMode = {};   // Pitch mode per sound (1.0 = normal, 1.5 = chipmunk)
let soundNames = {};       // Custom display names
let stopAllKeybind = '';
let isRecordingKeybind = false;
let isRecordingStopKeybind = false;
let micEnabled = false;
let youtubePlayingInterval = null;
let playingCheckInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initIcons();
    await ();
    await refreshSounds();
    updateStopKeybindUI();
    setupKeyboardListener();
    setupDragDrop();
    await initMicStatus();
    startPlayingCheck();
});

// Initialize icons
function initIcons() {
    document.getElementById('title-icon').innerHTML = Icons.music;
    document.getElementById('btn-add').innerHTML = Icons.add;
    document.getElementById('btn-refresh').innerHTML = Icons.refresh;
    document.getElementById('stop-icon').innerHTML = Icons.stop;

    const placeholder = document.getElementById('placeholder-icon');
    if (placeholder) placeholder.innerHTML = Icons.waveform;
}

// Load saved settings
async function loadSettings() {
    try {
        const settings = await eel.get_settings()();
        soundVolumes = settings.volumes || {};
        soundKeybinds = settings.keybinds || {};
        soundScreamMode = settings.screamMode || {};
        soundPitchMode = settings.pitchMode || {};
        soundNames = settings.names || {};
        stopAllKeybind = settings.stopAllKeybind || '';
    } catch (e) {
        console.log('No saved settings');
    }
}

// Save settings
async function saveSettings() {
    try {
        await eel.save_settings({
            volumes: soundVolumes,
            keybinds: soundKeybinds,
            screamMode: soundScreamMode,
            pitchMode: soundPitchMode,
            names: soundNames,
            stopAllKeybind: stopAllKeybind
        })();
    } catch (e) {
        console.error('Error saving settings:', e);
    }
}

// Update Stop All keybind UI
function updateStopKeybindUI() {
    const el = document.getElementById('stop-keybind');
    const textEl = document.getElementById('stop-keybind-text');
    if (el && textEl) {
        if (stopAllKeybind) {
            el.classList.add('has-bind');
            textEl.textContent = stopAllKeybind;
        } else {
            el.classList.remove('has-bind');
            textEl.textContent = 'Add keybind';
        }
    }
}

// Start recording Stop All keybind
function startStopKeybindRecord() {
    isRecordingStopKeybind = true;
    isRecordingKeybind = false;

    const el = document.getElementById('stop-keybind');
    const textEl = document.getElementById('stop-keybind-text');
    if (el && textEl) {
        el.classList.add('recording');
        textEl.textContent = 'Press a key...';
    }
}

// Refresh sound list
async function refreshSounds() {
    try {
        const sounds = await eel.get_sounds()();
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

        grid.innerHTML = sounds.map(name => {
            const keybind = soundKeybinds[name] || '';
            const isScream = soundScreamMode[name] || false;
            const isPitch = soundPitchMode[name] || false;
            const displayName = soundNames[name] || name;
            const badges = [];
            if (isScream) badges.push('😈');
            if (isPitch) badges.push('🐿️');
            return `
                <div class="sound-card ${isScream ? 'scream-mode' : ''} ${isPitch ? 'pitch-mode' : ''}" data-name="${escapeAttr(name)}">
                    <div class="sound-thumbnail">
                        <span class="thumb-icon">${Icons.waveform}</span>
                        ${badges.length ? `<span class="mode-badges">${badges.join('')}</span>` : ''}
                    </div>
                    <div class="sound-name" title="${escapeAttr(displayName)}">${escapeHtml(displayName)}</div>
                    <div class="sound-keybind ${keybind ? 'has-bind' : ''}">
                        ${keybind || 'Add keybind'}
                    </div>
                </div>
            `;
        }).join('');

        // Add event listeners using event delegation
        setupSoundCardEvents();
    } catch (error) {
        console.error('Error refreshing sounds:', error);
    }
}

// Select sound and show panel
function selectSound(name) {
    document.querySelectorAll('.sound-card').forEach(card => {
        card.classList.remove('selected');
        if (card.dataset.name === name) {
            card.classList.add('selected');
        }
    });

    selectedSound = name;
    showSoundPanel(name);
}

// Show right panel for selected sound
function showSoundPanel(name) {
    const panel = document.getElementById('right-panel');
    const volume = soundVolumes[name] !== undefined ? soundVolumes[name] : 100;
    const keybind = soundKeybinds[name] || '';
    const isScream = soundScreamMode[name] || false;
    const displayName = soundNames[name] || name;

    panel.innerHTML = `
        <div class="panel-header">
            <input type="text" class="panel-sound-name editable" id="sound-name-input" 
                   value="${escapeAttr(displayName)}" 
                   placeholder="${escapeAttr(name)}"
                   onchange="onSoundNameChange(this.value)"
                   onkeydown="if(event.key==='Enter')this.blur()">
            <div class="panel-sound-info">${escapeHtml(name)}</div>
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
                   onclick="startKeybindRecordPanel()">
        </div>
        
        <div class="panel-section">
            <div class="panel-section-title">${icon('volume', 16)} Volume</div>
            <div class="volume-control">
                <input type="range" class="volume-slider" id="sound-volume" 
                       min="0" max="100" value="${volume}"
                       oninput="onSoundVolumeChange(this.value)">
                <span class="volume-value" id="volume-value">${volume}</span>
            </div>
        </div>
        
        <div class="panel-section">
            <div class="panel-section-title">😈 Scream Mode</div>
            <label class="scream-toggle">
                <input type="checkbox" id="scream-checkbox" ${isScream ? 'checked' : ''} onchange="toggleScreamMode()">
                <span class="scream-slider"></span>
                <span class="scream-label">${isScream ? 'ON - 5000% BOOST! 💀' : 'OFF'}</span>
            </label>
            <div class="scream-hint">Boost volume to max for trolling</div>
        </div>
        
        <div class="panel-section">
            <div class="panel-section-title">🐿️ Chipmunk Mode</div>
            <label class="pitch-toggle">
                <input type="checkbox" id="pitch-checkbox" ${soundPitchMode[name] ? 'checked' : ''} onchange="togglePitchMode()">
                <span class="pitch-slider"></span>
                <span class="pitch-label">${soundPitchMode[name] ? 'ON - HIGH PITCH!' : 'OFF'}</span>
            </label>
            <div class="pitch-hint">Speed up audio for chipmunk voice</div>
        </div>
        
        <div class="panel-actions">
            <button class="btn-panel btn-play" onclick="playSound('${escapeAttr(name)}')">${icon('play', 14)} Play</button>
            <button class="btn-panel btn-stop" onclick="stopAll()">${icon('stop', 14)} Stop</button>
        </div>
        
        <div class="panel-actions">
            <button class="btn-panel btn-delete" onclick="deleteSound('${escapeAttr(name)}')">${icon('trash', 14)} Delete</button>
        </div>
    `;
}

// Toggle scream mode
function toggleScreamMode() {
    if (!selectedSound) return;
    const checkbox = document.getElementById('scream-checkbox');
    const isScream = checkbox.checked;
    soundScreamMode[selectedSound] = isScream;
    saveSettings();

    // Update label
    const label = document.querySelector('.scream-label');
    if (label) label.textContent = isScream ? 'ON - 5000% BOOST! 💀' : 'OFF';

    // Update wave animation
    const wave = document.querySelector('.preview-wave');
    if (wave) {
        if (isScream) wave.classList.add('scream-active');
        else wave.classList.remove('scream-active');
    }

    // Update card indicator
    refreshSounds().then(() => selectSound(selectedSound));
}

// Toggle pitch/chipmunk mode
function togglePitchMode() {
    if (!selectedSound) return;
    const checkbox = document.getElementById('pitch-checkbox');
    const isPitch = checkbox.checked;
    soundPitchMode[selectedSound] = isPitch;
    saveSettings();

    // Update label
    const label = document.querySelector('.pitch-label');
    if (label) label.textContent = isPitch ? 'ON - HIGH PITCH!' : 'OFF';

    // Update card indicator
    refreshSounds().then(() => selectSound(selectedSound));
}

// Check playing status periodically
function startPlayingCheck() {
    if (playingCheckInterval) {
        clearInterval(playingCheckInterval);
    }

    playingCheckInterval = setInterval(async () => {
        try {
            const playingSound = await eel.get_playing_sound()();

            // Update UI
            document.querySelectorAll('.sound-card').forEach(card => {
                if (card.dataset.name === playingSound) {
                    card.classList.add('playing');
                } else {
                    card.classList.remove('playing');
                }
            });
        } catch (e) {
            // Ignore errors
        }
    }, 100); // Check every 100ms for smooth animation
}

// Play sound
async function playSound(name) {
    try {
        // Immediately add playing class
        document.querySelectorAll('.sound-card').forEach(card => {
            if (card.dataset.name === name) {
                card.classList.add('playing');
            }
        });

        let volume = (soundVolumes[name] !== undefined ? soundVolumes[name] : 100) / 100;
        const isScream = soundScreamMode[name] || false;
        const isPitch = soundPitchMode[name] || false;
        if (isScream) volume = Math.min(volume * 50.0, 50.0);  // 5000% boost 💀
        const pitch = isPitch ? 1.5 : 1.0;  // Chipmunk = 1.5x speed

        await eel.play_sound(name, volume, pitch)();
    } catch (error) {
        console.error('Error playing sound:', error);
    }
}

// Stop all sounds
async function stopAll() {
    try {
        await eel.stop_all()();
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.remove('playing');
        });
    } catch (error) {
        console.error('Error stopping sounds:', error);
    }
}

// Sound volume change
function onSoundVolumeChange(value) {
    if (!selectedSound) return;
    document.getElementById('volume-value').textContent = value;
    soundVolumes[selectedSound] = parseInt(value);
    saveSettings();
}

// Sound name change
function onSoundNameChange(value) {
    if (!selectedSound) return;
    const newName = value.trim();
    if (newName && newName !== selectedSound) {
        soundNames[selectedSound] = newName;
    } else {
        delete soundNames[selectedSound];
    }
    saveSettings();
    refreshSounds().then(() => selectSound(selectedSound));
}

// Keybind recording for sounds
function startKeybindRecord(name) {
    selectedSound = name;
    selectSound(name);
    startKeybindRecordPanel();
}

function startKeybindRecordPanel() {
    if (!selectedSound) return;
    const input = document.getElementById('keybind-input');
    if (!input) return;

    isRecordingKeybind = true;
    isRecordingStopKeybind = false;
    input.classList.add('recording');
    input.value = 'Press a key...';
    input.focus();
}

// Get key name from event.code
function getKeyFromCode(code) {
    if (code.startsWith('Digit')) return code.replace('Digit', '');
    if (code.startsWith('Numpad')) return 'Num' + code.replace('Numpad', '');
    if (code.startsWith('Key')) return code.replace('Key', '');
    if (code.startsWith('F') && !isNaN(code.slice(1))) return code;

    const specialKeys = {
        'Space': 'Space', 'Enter': 'Enter', 'Escape': 'Esc',
        'Backspace': 'Backspace', 'Tab': 'Tab',
        'ArrowUp': 'Up', 'ArrowDown': 'Down', 'ArrowLeft': 'Left', 'ArrowRight': 'Right',
        'Delete': 'Delete', 'Insert': 'Insert', 'Home': 'Home', 'End': 'End',
        'PageUp': 'PageUp', 'PageDown': 'PageDown',
        'Minus': '-', 'Equal': '=', 'BracketLeft': '[', 'BracketRight': ']',
        'Backslash': '\\', 'Semicolon': ';', 'Quote': "'",
        'Comma': ',', 'Period': '.', 'Slash': '/', 'Backquote': '`'
    };
    return specialKeys[code] || code;
}

// Check if key is a modifier
function isModifierKey(code) {
    return ['ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight', 'AltLeft', 'AltRight', 'MetaLeft', 'MetaRight'].includes(code);
}

// Build keybind string from event
function buildKeybindString(e) {
    // Otherwise build combo
    let keybind = '';
    if (e.ctrlKey) keybind += 'Ctrl + ';
    if (e.shiftKey) keybind += 'Shift + ';
    if (e.altKey) keybind += 'Alt + ';
    keybind += getKeyFromCode(e.code);
    return keybind;
}

// Keyboard listener
function setupKeyboardListener() {
    document.addEventListener('keydown', (e) => {
        // Recording Stop All keybind - skip if only modifier key pressed
        if (isRecordingStopKeybind) {
            e.preventDefault();

            // Skip if only modifier key (wait for actual key)
            if (isModifierKey(e.code)) {
                return;
            }

            stopAllKeybind = buildKeybindString(e);
            saveSettings();

            const el = document.getElementById('stop-keybind');
            if (el) el.classList.remove('recording');
            updateStopKeybindUI();

            isRecordingStopKeybind = false;
            return;
        }

        // Recording sound keybind - skip if only modifier key pressed
        if (isRecordingKeybind && selectedSound) {
            e.preventDefault();

            // Skip if only modifier key (wait for actual key)
            if (isModifierKey(e.code)) {
                return;
            }

            const keybind = buildKeybindString(e);
            soundKeybinds[selectedSound] = keybind;
            saveSettings();

            const input = document.getElementById('keybind-input');
            if (input) {
                input.value = keybind;
                input.classList.remove('recording');
            }

            refreshSounds().then(() => selectSound(selectedSound));
            isRecordingKeybind = false;
            return;
        }

        // Check Stop All keybind
        if (stopAllKeybind && matchKeybind(e, stopAllKeybind)) {
            e.preventDefault();
            stopAll();
            return;
        }

        // Check sound keybinds
        for (const [name, bind] of Object.entries(soundKeybinds)) {
            if (matchKeybind(e, bind)) {
                e.preventDefault();
                playSound(name);
                return;
            }
        }
    });
}

// Match keybind
function matchKeybind(event, keybind) {
    if (!keybind) return false;

    // Single modifier key (Alt, Shift, Ctrl alone)
    if (keybind === 'Alt') {
        return event.code.startsWith('Alt') && !event.ctrlKey && !event.shiftKey;
    }
    if (keybind === 'Shift') {
        return event.code.startsWith('Shift') && !event.ctrlKey && !event.altKey;
    }
    if (keybind === 'Ctrl') {
        return event.code.startsWith('Control') && !event.shiftKey && !event.altKey;
    }

    // Combo keybind
    const parts = keybind.split(' + ');
    const key = parts[parts.length - 1];
    const needShift = parts.includes('Shift');
    const needCtrl = parts.includes('Ctrl');
    const needAlt = parts.includes('Alt');

    if (needShift !== event.shiftKey) return false;
    if (needCtrl !== event.ctrlKey) return false;
    if (needAlt !== event.altKey) return false;

    return getKeyFromCode(event.code) === key;
}

// Add sound
async function addSound() {
    try {
        const result = await eel.add_sound_dialog()();
        if (result) await refreshSounds();
    } catch (error) {
        console.error('Error adding sound:', error);
    }
}

// Delete sound
async function deleteSound(name) {
    if (!confirm(`Delete "${name}"?`)) return;

    try {
        await eel.delete_sound(name)();
        delete soundVolumes[name];
        delete soundKeybinds[name];
        saveSettings();

        selectedSound = null;
        document.getElementById('right-panel').innerHTML = `
            <div class="panel-placeholder">
                <span class="placeholder-icon">${Icons.waveform}</span>
                <p>Click a sound to edit</p>
            </div>
        `;

        await refreshSounds();
    } catch (error) {
        console.error('Error deleting sound:', error);
    }
}

// Helpers
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    // Escape for HTML attributes
    return text
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

// Setup event listeners for sound cards (handles special characters in names)
function setupSoundCardEvents() {
    const grid = document.getElementById('sounds-grid');

    // Remove old listeners by cloning
    const newGrid = grid.cloneNode(true);
    grid.parentNode.replaceChild(newGrid, grid);

    // Single click - select
    newGrid.addEventListener('click', (e) => {
        const card = e.target.closest('.sound-card');
        if (!card) return;

        const name = card.dataset.name;

        // Check if clicked on keybind
        if (e.target.closest('.sound-keybind')) {
            e.stopPropagation();
            startKeybindRecord(name);
            return;
        }

        selectSound(name);
    });

    // Double click - play
    newGrid.addEventListener('dblclick', (e) => {
        const card = e.target.closest('.sound-card');
        if (!card) return;

        const name = card.dataset.name;
        playSound(name);
    });
}


// Drag & Drop support
function setupDragDrop() {
    const app = document.querySelector('.app');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
        app.addEventListener(event, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    // Highlight on drag
    ['dragenter', 'dragover'].forEach(event => {
        app.addEventListener(event, () => {
            app.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(event => {
        app.addEventListener(event, () => {
            app.classList.remove('drag-over');
        });
    });

    // Handle drop
    app.addEventListener('drop', async (e) => {
        const files = e.dataTransfer.files;
        if (files.length === 0) return;

        // Filter and read audio files
        let addedCount = 0;
        for (const file of files) {
            if (file.type.startsWith('audio/') ||
                /\.(wav|mp3|ogg|flac|m4a)$/i.test(file.name)) {
                try {
                    // Read file as base64
                    const base64 = await readFileAsBase64(file);
                    const added = await eel.add_sound_base64(file.name, base64)();
                    if (added) addedCount++;
                } catch (err) {
                    console.error('Error reading file:', err);
                }
            }
        }

        if (addedCount === 0) {
            alert('No audio files added. Supported: WAV, MP3, OGG, FLAC');
        } else {
            await refreshSounds();
        }
    });
}

// Read file as base64
function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // Remove data URL prefix
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}


// Mic Passthrough
async function initMicStatus() {
    try {
        micEnabled = await eel.is_mic_enabled()();
        updateMicUI();
    } catch (e) {
        console.log('Mic status check failed');
    }
}

function updateMicUI() {
    const btn = document.getElementById('btn-mic');
    const text = document.getElementById('mic-text');
    if (btn && text) {
        if (micEnabled) {
            btn.classList.add('active');
            text.textContent = 'Mic ON';
        } else {
            btn.classList.remove('active');
            text.textContent = 'Mic OFF';
        }
    }
}

async function toggleMic() {
    try {
        micEnabled = !micEnabled;
        await eel.toggle_mic_passthrough(micEnabled)();
        updateMicUI();
    } catch (error) {
        console.error('Error toggling mic:', error);
        micEnabled = !micEnabled;  // Revert
    }
}

async function onMicVolumeChange(value) {
    try {
        const vol = parseInt(value) / 100;  // 0-2.0
        await eel.set_mic_volume(vol)();
    } catch (error) {
        console.error('Error setting mic volume:', error);
    }
}


// YouTube Streaming
async function playYoutube() {
    const urlInput = document.getElementById('youtube-url');
    const url = urlInput.value.trim();

    if (!url) {
        alert('Please enter a YouTube URL');
        return;
    }

    const infoEl = document.getElementById('youtube-info');
    const playBtn = document.getElementById('btn-youtube-play');

    infoEl.textContent = 'Loading...';
    infoEl.className = 'youtube-info loading';
    playBtn.disabled = true;

    try {
        const result = await eel.play_youtube(url)();

        if (result.success) {
            infoEl.textContent = '▶ ' + result.title;
            infoEl.className = 'youtube-info playing';
            startYoutubeStatusCheck();
        } else {
            infoEl.textContent = '❌ ' + (result.error || 'Failed to play');
            infoEl.className = 'youtube-info error';
        }
    } catch (error) {
        console.error('YouTube error:', error);
        infoEl.textContent = '❌ Error: ' + error;
        infoEl.className = 'youtube-info error';
    } finally {
        playBtn.disabled = false;
    }
}

async function stopYoutube() {
    try {
        await eel.stop_youtube()();
        updateYoutubeUI(false);
    } catch (error) {
        console.error('Error stopping YouTube:', error);
    }
}

function updateYoutubeUI(playing, title = '') {
    const infoEl = document.getElementById('youtube-info');
    if (playing && title) {
        infoEl.textContent = '▶ ' + title;
        infoEl.className = 'youtube-info playing';
    } else {
        infoEl.textContent = '';
        infoEl.className = 'youtube-info';
    }
}

function startYoutubeStatusCheck() {
    // Clear existing interval
    if (youtubePlayingInterval) {
        clearInterval(youtubePlayingInterval);
    }

    // Check status every 2 seconds
    youtubePlayingInterval = setInterval(async () => {
        try {
            const info = await eel.get_youtube_info()();
            if (!info.playing) {
                updateYoutubeUI(false);
                clearInterval(youtubePlayingInterval);
                youtubePlayingInterval = null;
            }
        } catch (e) {
            clearInterval(youtubePlayingInterval);
            youtubePlayingInterval = null;
        }
    }, 2000);
}

async function onYoutubeVolumeChange(value) {
    try {
        const vol = parseInt(value) / 100;  // 0-2.0
        await eel.set_youtube_volume(vol)();
    } catch (error) {
        console.error('Error setting YouTube volume:', error);
    }
}
