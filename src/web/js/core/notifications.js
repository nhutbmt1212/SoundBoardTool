const Notifications = {
    container: null,

    init() {
        this.container = document.createElement('div');
        this.container.className = 'notifications-container';
        document.body.appendChild(this.container);
    },

    show(message, type = 'info') {
        if (!this.container) this.init();

        const toast = document.createElement('div');
        toast.className = `notification toast-${type}`;

        let iconName = 'info';
        if (type === 'success') iconName = 'check';
        if (type === 'error' || type === 'warning') iconName = 'warning';

        const iconSvg = IconManager.get(iconName, { size: 20 });

        toast.innerHTML = `
            <div class="notification-icon">${iconSvg}</div>
            <div class="notification-content">${message}</div>
            <div class="notification-close" onclick="this.parentElement.remove()">${IconManager.get('close', { size: 16 })}</div>
        `;

        this.container.appendChild(toast);

        // Animate in
        requestAnimationFrame(() => toast.classList.add('show'));

        // Auto remove
        setTimeout(() => {
            toast.classList.remove('show');
            toast.addEventListener('transitionend', () => toast.remove());
        }, 3000);
    },

    // Shortcuts
    success(msg) { this.show(msg, 'success'); },
    error(msg) { this.show(msg, 'error'); },
    warning(msg) { this.show(msg, 'warning'); },
    info(msg) { this.show(msg, 'info'); }
};

window.Notifications = Notifications;
