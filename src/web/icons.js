// Font Awesome Icons Map
const Icons = {
    // App
    music: 'fa-solid fa-music',

    // Controls
    play: 'fa-solid fa-play',
    stop: 'fa-solid fa-stop',
    pause: 'fa-solid fa-pause',

    // Actions
    add: 'fa-solid fa-plus',
    refresh: 'fa-solid fa-rotate',
    trash: 'fa-solid fa-trash',

    // Audio
    volume: 'fa-solid fa-volume-high',
    volumeMute: 'fa-solid fa-volume-xmark',

    // Keyboard
    keyboard: 'fa-solid fa-keyboard',

    // Mic
    mic: 'fa-solid fa-microphone',
    micOff: 'fa-solid fa-microphone-slash',

    // Status
    check: 'fa-solid fa-check',
    warning: 'fa-solid fa-triangle-exclamation',
    info: 'fa-solid fa-circle-info',

    // Waveform/Visuals
    waveform: 'fa-solid fa-wave-square',
    folder: 'fa-solid fa-folder',
    heart: 'fa-solid fa-heart',

    // Settings
    settings: 'fa-solid fa-gear',

    // Effects
    scream: 'fa-solid fa-skull', // or fa-ghost
    chipmunk: 'fa-solid fa-wind', // Indicates speed/high pitch

    // Indicators
    playCircle: 'fa-solid fa-circle-play',
    pauseCircle: 'fa-solid fa-circle-pause',

    // Brands
    youtube: 'fa-brands fa-youtube',

    // UI
    close: 'fa-solid fa-xmark',
};

// Smart Icon Manager
const IconManager = {
    // get returns the HTML string for the icon
    get(name, { size = 20, cls = '', color = '' } = {}) {
        const iconClass = Icons[name] || Icons.music;

        // Font Awesome sizing can be handled via font-size or fa- classes
        // We'll use inline styles for precise pixel control to match previous logic
        let style = `font-size: ${size}px;`;
        if (color) style += `color: ${color};`;

        return `<i class="${iconClass} ${cls}" style="${style}"></i>`;
    }
};

// Export for use
window.Icons = Icons;
window.IconManager = IconManager;
// Deprecated simple helper, kept for compatibility if needed, but mapped to Manager
window.icon = (name, size, cls) => IconManager.get(name, { size, cls });
