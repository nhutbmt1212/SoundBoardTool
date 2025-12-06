// Soundboard Pro - Frontend Logic

let selectedSound = null;
let soundVolumes = {};
let soundKeybinds = {};
let stopAllKeybind = '';
let isRecordingKeybind = false;
let isRecordingStopKeybind = false;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initIcons();
    await loadSettings();
    await refreshSounds();
    updateStopKeybindUI();
    setupKeyboardListener();
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
            return `
                <div class="sound-card" data-name="${escapeAttr(name)}" onclick="selectSound('${escapeAttr(name)}')" ondblclick="playSound('${escapeAttr(name)}')">
                    <div class="sound-thumbnail">
                        <span class="thumb-icon">${Icons.waveform}</span>
                    </div>
                    <div class="sound-name" title="${escapeAttr(name)}">${escapeHtml(name)}</div>
                    <div class="sound-keybind ${keybind ? 'has-bind' : ''}" onclick="event.stopPropagation(); startKeybindRecord('${escapeAttr(name)}')">
                        ${keybind || 'Add keybind'}
                    </div>
                </div>
            `;
        }).join('');
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
    
    panel.innerHTML = `
        <div class="panel-header">
            <div class="panel-sound-name">${escapeHtml(name)}</div>
            <div class="panel-sound-info">Audio file</div>
        </div>
        
        <div class="panel-preview">
            <div class="preview-wave">
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
        
        <div class="panel-actions">
            <button class="btn-panel btn-play" onclick="playSound('${escapeAttr(name)}')">${icon('play', 14)} Play</button>
            <button class="btn-panel btn-stop" onclick="stopAll()">${icon('stop', 14)} Stop</button>
        </div>
        
        <div class="panel-actions">
            <button class="btn-panel btn-delete" onclick="deleteSound('${escapeAttr(name)}')">${icon('trash', 14)} Delete</button>
        </div>
    `;
}

// Play sound
async function playSound(name) {
    try {
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.remove('playing');
            if (card.dataset.name === name) {
                card.classList.add('playing');
            }
        });
        
        const volume = (soundVolumes[name] !== undefined ? soundVolumes[name] : 100) / 100;
        await eel.play_sound(name, volume)();
        
        setTimeout(() => {
            document.querySelectorAll('.sound-card').forEach(card => {
                card.classList.remove('playing');
            });
        }, 500);
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

// Build keybind string from event
function buildKeybindString(e) {
    const modifierCodes = ['ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight', 'AltLeft', 'AltRight', 'MetaLeft', 'MetaRight'];
    const isModifierOnly = modifierCodes.includes(e.code);
    
    // If it's a modifier key alone, just return that modifier
    if (isModifierOnly) {
        if (e.code.startsWith('Shift')) return 'Shift';
        if (e.code.startsWith('Control')) return 'Ctrl';
        if (e.code.startsWith('Alt')) return 'Alt';
        return e.code;
    }
    
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
        // Recording Stop All keybind - accept ANY key including modifiers
        if (isRecordingStopKeybind) {
            e.preventDefault();
            
            stopAllKeybind = buildKeybindString(e);
            saveSettings();
            
            const el = document.getElementById('stop-keybind');
            if (el) el.classList.remove('recording');
            updateStopKeybindUI();
            
            isRecordingStopKeybind = false;
            return;
        }
        
        // Recording sound keybind - accept ANY key including modifiers
        if (isRecordingKeybind && selectedSound) {
            e.preventDefault();
            
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
    return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
