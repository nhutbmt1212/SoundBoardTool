// Soundboard Pro - Frontend Logic

// Initialize when page loads
document.addEventListener('DOMContentLoaded', async () => {
    await refreshSounds();
    await updateStatus();
    
    // Set initial volume
    const volume = await eel.get_volume()();
    document.getElementById('volume').value = volume * 100;
    document.getElementById('volume-value').textContent = `${Math.round(volume * 100)}%`;
});

// Refresh sound list
async function refreshSounds() {
    try {
        const sounds = await eel.get_sounds()();
        const grid = document.getElementById('sounds-grid');
        
        if (sounds.length === 0) {
            grid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <h2>📂 No sounds found</h2>
                    <p>Click 'Add Sound' to get started!</p>
                </div>
            `;
            return;
        }
        
        grid.innerHTML = sounds.map(name => `
            <button class="btn-sound" onclick="playSound('${escapeHtml(name)}')">
                ${escapeHtml(name)}
            </button>
        `).join('');
    } catch (error) {
        console.error('Error refreshing sounds:', error);
    }
}

// Play sound
async function playSound(name) {
    try {
        // Visual feedback
        const buttons = document.querySelectorAll('.btn-sound');
        buttons.forEach(btn => {
            if (btn.textContent.trim() === name) {
                btn.classList.add('playing');
                setTimeout(() => btn.classList.remove('playing'), 500);
            }
        });
        
        await eel.play_sound(name)();
    } catch (error) {
        console.error('Error playing sound:', error);
    }
}

// Stop all sounds
async function stopAll() {
    try {
        await eel.stop_all()();
    } catch (error) {
        console.error('Error stopping sounds:', error);
    }
}

// Volume change
async function onVolumeChange(value) {
    document.getElementById('volume-value').textContent = `${value}%`;
    try {
        await eel.set_volume(parseInt(value) / 100)();
    } catch (error) {
        console.error('Error setting volume:', error);
    }
}

// Add sound
async function addSound() {
    try {
        const result = await eel.add_sound_dialog()();
        if (result) {
            await refreshSounds();
            showNotification('✅ Sound added successfully!');
        }
    } catch (error) {
        console.error('Error adding sound:', error);
    }
}

// Update VB-Cable status
async function updateStatus() {
    try {
        const connected = await eel.is_vb_cable_connected()();
        const statusEl = document.getElementById('status');
        
        if (connected) {
            statusEl.textContent = "🎙️ Discord: Chọn 'CABLE Output' làm Input";
            statusEl.className = 'status connected';
        } else {
            statusEl.textContent = "⚠️ VB-Cable chưa cài - Tải tại vb-audio.com/Cable";
            statusEl.className = 'status warning';
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Helper: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Helper: Show notification
function showNotification(message) {
    // Simple alert for now, can be replaced with toast
    alert(message);
}
