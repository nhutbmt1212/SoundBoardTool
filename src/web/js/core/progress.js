// Expose function to receive progress from backend
window.onYoutubeProgress = function (url, percent) {
    if (window.UI && window.UI.updateYoutubeProgress) {
        window.UI.updateYoutubeProgress(url, percent);
    }
};

// Make it available to eel
if (typeof eel !== 'undefined') {
    eel.expose(onYoutubeProgress, 'onYoutubeProgress');
}
