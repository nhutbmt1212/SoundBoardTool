// Card Renderer - Unified card/item rendering for all media types
// Eliminates duplication between Sound, YouTube, and TikTok cards

const CardRenderer = {
    /**
     * Render a sound card
     * @param {string} name - Sound file name
     */
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

    /**
     * Render a stream item (YouTube or TikTok)
     * @param {string} type - 'youtube' or 'tiktok'
     * @param {Object} item - Item data {url, title}
     * @param {Object} info - Playback info {url, playing, paused}
     */
    renderStreamItem(type, item, info) {
        const getKeybind = type === 'youtube' ? AppState.getYoutubeKeybind : AppState.getTikTokKeybind;
        const getDisplayName = type === 'youtube' ? AppState.getYoutubeDisplayName : AppState.getTikTokDisplayName;

        const keybind = getKeybind.call(AppState, item.url) || item.keybind || '';
        const displayName = getDisplayName.call(AppState, item.url, item.title);
        const isCurrentUrl = info && info.url === item.url;
        const isPlaying = isCurrentUrl && info.playing;
        const isPaused = isCurrentUrl && info.paused;
        const className = type === 'youtube' ? 'youtube-item' : 'tiktok-item';

        return `
            <div class="sound-card ${className} ${isPlaying ? 'playing' : ''} ${isPaused ? 'paused' : ''}" data-url="${Utils.escapeAttr(item.url)}">
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

    /**
     * Add a loading card to grid
     * @param {string} type - 'youtube' or 'tiktok'
     * @param {string} url - URL being downloaded
     */
    addLoadingCard(type, url) {
        const gridId = type === 'youtube' ? 'youtube-grid' : 'tiktok-grid';
        const grid = document.getElementById(gridId);
        if (!grid) return;

        // Remove empty state
        const empty = grid.querySelector('.empty-state');
        if (empty) empty.remove();

        const card = document.createElement('div');
        const className = type === 'youtube' ? 'youtube-item' : 'tiktok-item';
        card.className = `sound-card ${className} loading`;
        card.dataset.url = 'loading-' + url;
        card.innerHTML = UIHelpers.buildLoadingCard({ url });

        grid.prepend(card);
    },

    /**
     * Update loading progress
     * @param {string} type - 'youtube' or 'tiktok'
     * @param {string} url - URL being downloaded
     * @param {number} percent - Progress percentage (0-100)
     */
    updateProgress(type, url, percent) {
        const gridId = type === 'youtube' ? 'youtube-grid' : 'tiktok-grid';
        const className = type === 'youtube' ? 'youtube-item' : 'tiktok-item';
        const grid = document.getElementById(gridId);
        if (!grid) return;

        const card = grid.querySelector(`.${className}[data-url="loading-${Utils.escapeAttr(url)}"]`);
        if (!card) return;

        const pie = card.querySelector('.loading-pie');
        const text = card.querySelector('.loading-percent');

        if (pie) pie.style.setProperty('--p', percent);
        if (text) text.textContent = Math.round(percent) + '%';
    },

    /**
     * Remove loading card from grid
     * @param {string} type - 'youtube' or 'tiktok'
     * @param {string} url - URL that finished downloading
     */
    removeLoadingCard(type, url) {
        const gridId = type === 'youtube' ? 'youtube-grid' : 'tiktok-grid';
        const className = type === 'youtube' ? 'youtube-item' : 'tiktok-item';
        const grid = document.getElementById(gridId);
        if (!grid) return;

        const card = grid.querySelector(`.${className}[data-url="loading-${Utils.escapeAttr(url)}"]`);
        if (card) card.remove();
    }
};

window.CardRenderer = CardRenderer;
