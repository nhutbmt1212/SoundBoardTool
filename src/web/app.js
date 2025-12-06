// Soundboard Pro - Frontend Logic

let selectedSound = null;
let soundVolumes = {}; // Store individual sound volumes
let soundKeybinds = {}; // Store keybinds
let isRecordingKeybind = false;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadSettings();
    await refreshSounds();
    setupKeyboardListener();
});

// Load saved settings
async function loadSettings() {
    try {
        const settings = await eel.get_settings()();
        soundVolumes = settings.volumes || {};
        soundKeybinds = settings.keybinds || {};
    } catch (e) {
        console.log('No saved settings');
    }
}

// Save settings
async function saveSettings() {
    try {
        await eel.save_settings({
            volumes: soundVolumes,
            keybinds: soundKeybinds
        })();
    } catch (e) {
        console.error('Error saving settings:', e);
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
                    <h2>📂 No sounds found</h2>
                    <p>Click ➕ to add sounds</p>
                </div>
            `;
            return;
        }
        
        grid.innerHTML = sounds.map(name => {
            const keybind = soundKeybinds[name] || '';
            return `
                <div class="sound-card" data-name="${escapeAttr(name)}" onclick="selectSound('${escapeAttr(name)}')" ondblclick="playSound('${escapeAttr(name)}')">
                    <div class="sound-thumbnail">
                        <div class="sound-wave"></div>
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
    // Update selection UI
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
            <div class="panel-section-title">⌨️ Keybind</div>
            <input type="text" class="keybind-input" id="keybind-input" 
                   value="${keybind}" 
                   placeholder="Click to set keybind"
                   readonly
                   onclick="startKeybindRecordPanel()">
        </div>
        
        <div class="panel-section">
            <div class="panel-section-title">🔊 Volume</div>
            <div class="volume-control">
                <input type="range" class="volume-slider" id="sound-volume" 
                       min="0" max="100" value="${volume}"
                       oninput="onSoundVolumeChange(this.value)">
                <span class="volume-value" id="volume-value">${volume}</span>
            </div>
        </div>
        
        <div class="panel-actions">
            <button class="btn-panel btn-play" onclick="playSound('${escapeAttr(name)}')">▶ Play</button>
            <button class="btn-panel btn-stop" onclick="stopAll()">⏹ Stop</button>
        </div>
        
        <div class="panel-actions">
            <button class="btn-panel btn-delete" onclick="deleteSound('${escapeAttr(name)}')">🗑 Delete</button>
        </div>
    `;
}

// Play sound
async function playSound(name) {
    try {
        // Visual feedback
        document.querySelectorAll('.sound-card').forEach(card => {
            card.classList.remove('playing');
            if (card.dataset.name === name) {
                card.classList.add('playing');
            }
        });
        
        const volume = (soundVolumes[name] !== undefined ? soundVolumes[name] : 100) / 100;
        await eel.play_sound(name, volume)();
        
        // Remove playing class after a delay
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

// Keybind recording
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
    input.classList.add('recording');
    input.value = 'Press a key...';
    input.focus();
}

// Get key name from event.code (physical key)
function getKeyFromCode(code) {
    // Number keys
    if (code.startsWith('Digit')) return code.replace('Digit', '');
    if (code.startsWith('Numpad')) return 'Num' + code.replace('Numpad', '');
    
    // Letter keys
    if (code.startsWith('Key')) return code.replace('Key', '');
    
    // Function keys
    if (code.startsWith('F') && !isNaN(code.slice(1))) return code;
    
    // Special keys
    const specialKeys = {
        'Space': 'Space',
        'Enter': 'Enter',
        'Escape': 'Esc',
        'Backspace': 'Backspace',
        'Tab': 'Tab',
        'ArrowUp': 'Up',
        'ArrowDown': 'Down',
        'ArrowLeft': 'Left',
        'ArrowRight': 'Right',
        'Delete': 'Delete',
        'Insert': 'Insert',
        'Home': 'Home',
        'End': 'End',
        'PageUp': 'PageUp',
        'PageDown': 'PageDown',
        'Minus': '-',
        'Equal': '=',
        'BracketLeft': '[',
        'BracketRight': ']',
        'Backslash': '\\',
        'Semicolon': ';',
        'Quote': "'",
        'Comma': ',',
        'Period': '.',
        'Slash': '/',
        'Backquote': '`'
    };
    
    return specialKeys[code] || code;
}

// Keyboard listener
function setupKeyboardListener() {
    document.addEventListener('keydown', (e) => {
        // Ignore modifier-only keys
        const modifierCodes = ['ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight', 'AltLeft', 'AltRight', 'MetaLeft', 'MetaRight'];
        
        // Recording keybind
        if (isRecordingKeybind && selectedSound) {
            // Skip if only modifier key pressed
            if (modifierCodes.includes(e.code)) {
                return;
            }
            
            e.preventDefault();
            
            let keybind = '';
            if (e.ctrlKey) keybind += 'Ctrl + ';
            if (e.shiftKey) keybind += 'Shift + ';
            if (e.altKey) keybind += 'Alt + ';
            
            // Get key from code (physical key)
            const key = getKeyFromCode(e.code);
            keybind += key;
            
            // Save keybind
            soundKeybinds[selectedSound] = keybind;
            saveSettings();
            
            // Update UI
            const input = document.getElementById('keybind-input');
            if (input) {
                input.value = keybind;
                input.classList.remove('recording');
            }
            
            // Update card
            refreshSounds().then(() => selectSound(selectedSound));
            
            isRecordingKeybind = false;
            return;
        }
        
        // Play sound by keybind (skip if only modifier)
        if (modifierCodes.includes(e.code)) return;
        
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
    
    const parts = keybind.split(' + ');
    const key = parts[parts.length - 1];
    const needShift = parts.includes('Shift');
    const needCtrl = parts.includes('Ctrl');
    const needAlt = parts.includes('Alt');
    
    if (needShift !== event.shiftKey) return false;
    if (needCtrl !== event.ctrlKey) return false;
    if (needAlt !== event.altKey) return false;
    
    // Use code-based key matching
    const eventKey = getKeyFromCode(event.code);
    return eventKey === key;
}

// Add sound
async function addSound() {
    try {
        const result = await eel.add_sound_dialog()();
        if (result) {
            await refreshSounds();
        }
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
