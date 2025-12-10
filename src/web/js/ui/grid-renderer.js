// Grid Renderer - Unified grid rendering for all media types
// Eliminates duplication between Sound, YouTube, and TikTok grids

const GridRenderer = {
    /**
     * Render sound grid
     * @param {Array} sounds - Array of sound file names
     */
    renderSoundGrid(sounds) {
        const grid = document.getElementById('sounds-grid');

        if (sounds.length === 0) {
            grid.innerHTML = UIHelpers.buildEmptyState({
                iconName: 'folder',
                title: 'No sounds found',
                message: `Click ${icon('add', 16)} to add sounds`
            });
            return;
        }

        // Sort sounds alphabetically by display name
        const sortedSounds = [...sounds].sort((a, b) => {
            const nameA = AppState.getDisplayName(a).toLowerCase();
            const nameB = AppState.getDisplayName(b).toLowerCase();
            return nameA.localeCompare(nameB);
        });

        grid.innerHTML = sortedSounds.map(name => CardRenderer.renderSoundCard(name)).join('');
    },

    /**
     * Render stream grid (YouTube or TikTok)
     * @param {string} type - 'youtube' or 'tiktok'
     * @param {Array} items - Array of items {url, title}
     * @param {Object} info - Current playback info {url, playing, paused}
     */
    renderStreamGrid(type, items, info) {
        const gridId = type === 'youtube' ? 'youtube-grid' : 'tiktok-grid';
        const grid = document.getElementById(gridId);

        if (!items || items.length === 0) {
            const config = type === 'youtube'
                ? { iconName: 'youtube', title: 'No YouTube videos', message: 'Click "Add from YouTube" to get started' }
                : { iconName: 'tiktok', title: 'No TikTok videos', message: 'Click "Add from TikTok" to get started' };

            grid.innerHTML = UIHelpers.buildEmptyState(config);
            return;
        }

        // Sort items alphabetically by display name
        const getDisplayName = type === 'youtube' ? AppState.getYoutubeDisplayName : AppState.getTikTokDisplayName;
        const sortedItems = [...items].sort((a, b) => {
            const nameA = getDisplayName.call(AppState, a.url, a.title).toLowerCase();
            const nameB = getDisplayName.call(AppState, b.url, b.title).toLowerCase();
            return nameA.localeCompare(nameB);
        });

        grid.innerHTML = sortedItems.map(item => CardRenderer.renderStreamItem(type, item, info)).join('');
    }
};

window.GridRenderer = GridRenderer;
