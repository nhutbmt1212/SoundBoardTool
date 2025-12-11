// Backup Event Handlers for Google Drive Integration

/**
 * Backup event handlers module
 * Handles all backup-related user interactions
 */
const BackupEvents = {
    // ==================== State ====================

    /** @type {Object|null} Current backup status */
    currentStatus: null,

    // ==================== Initialization ====================

    /**
     * Initialize backup system
     * Load status and setup UI
     */
    async init() {
        await this.loadStatus();
        this.updateUI();
    },

    /**
     * Load current backup status from backend
     */
    async loadStatus() {
        try {
            this.currentStatus = await API.Backup.getStatus();
        } catch (error) {
            console.error('[BackupEvents] Failed to load status:', error);
            this.currentStatus = {
                is_logged_in: false,
                user_email: null,
                last_backup_time: null,
                auto_backup_enabled: false
            };
        }
    },

    /**
     * Update UI based on current status
     */
    updateUI() {
        const loginBtn = document.getElementById('backup-login-btn');
        const logoutBtn = document.getElementById('backup-logout-btn');
        const backupBtn = document.getElementById('backup-now-btn');
        const restoreBtn = document.getElementById('restore-btn');
        const autoBackupToggle = document.getElementById('auto-backup-toggle');
        const statusText = document.getElementById('backup-status-text');
        const userEmail = document.getElementById('backup-user-email');

        if (!this.currentStatus) return;

        const isLoggedIn = this.currentStatus.is_logged_in;

        // Update buttons visibility
        if (loginBtn) loginBtn.style.display = isLoggedIn ? 'none' : 'inline-flex';
        if (logoutBtn) logoutBtn.style.display = isLoggedIn ? 'inline-flex' : 'none';
        if (backupBtn) backupBtn.disabled = !isLoggedIn;
        if (restoreBtn) restoreBtn.disabled = !isLoggedIn;
        if (autoBackupToggle) {
            autoBackupToggle.disabled = !isLoggedIn;
            autoBackupToggle.checked = this.currentStatus.auto_backup_enabled;
        }

        // Update status text
        if (statusText) {
            if (isLoggedIn) {
                const lastBackup = this.currentStatus.last_backup_time
                    ? new Date(this.currentStatus.last_backup_time).toLocaleString()
                    : 'Never';
                statusText.textContent = `Last backup: ${lastBackup}`;
            } else {
                statusText.textContent = 'Not logged in';
            }
        }

        // Update user email
        if (userEmail) {
            userEmail.textContent = this.currentStatus.user_email || '';
            userEmail.style.display = isLoggedIn ? 'block' : 'none';
        }
    },

    // ==================== Login/Logout ====================

    /**
     * Show Google login - automatic flow with polling
     */
    async showLoginModal() {
        try {
            // Show loading modal
            const modalId = UI.showModal({
                title: 'Login to Google Drive',
                body: `
                    <div class="backup-login-modal">
                        <div class="login-status">
                            <div class="spinner"></div>
                            <p class="status-text">Opening browser...</p>
                            <p class="status-subtext">Please complete the authorization in your browser</p>
                        </div>
                        <div class="login-info">
                            <p>✓ Secure OAuth 2.0 authentication</p>
                            <p>✓ Your credentials are stored locally</p>
                            <p>✓ Browser will open automatically</p>
                        </div>
                    </div>
                `,
                showCancel: true,
                showConfirm: false,
                cancelText: 'Cancel'
            });

            // Start login flow
            const result = await API.Backup.startLogin();

            if (!result.success) {
                UI.closeModal(modalId);
                Notifications.error(`Failed to start login: ${result.error}`);
                return;
            }

            // Update modal to show waiting state
            const statusText = document.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = 'Waiting for authorization...';
            }

            // Poll for login status
            let pollCount = 0;
            const maxPolls = 240; // 2 minutes (240 * 500ms)

            const pollInterval = setInterval(async () => {
                pollCount++;

                // Check login status
                const loginResult = await eel.check_login_status()();

                if (loginResult) {
                    // Login completed (success or failure)
                    clearInterval(pollInterval);

                    if (loginResult.success) {
                        this.onLoginComplete(loginResult);
                    } else {
                        this.onLoginFailed(loginResult);
                    }
                } else if (pollCount >= maxPolls) {
                    // Timeout
                    clearInterval(pollInterval);
                    this.onLoginFailed({ error: 'Timeout waiting for authorization' });
                }
            }, 500); // Poll every 500ms

        } catch (error) {
            console.error('[BackupEvents] Failed to show login modal:', error);
            Notifications.error('Failed to start login process');
        }
    },

    /**
     * Handle successful login (called from backend)
     * @param {Object} result - Login result with email
     */
    onLoginComplete(result) {
        console.log('[BackupEvents] Login completed:', result);

        // Close loading modal properly
        const modal = document.getElementById('modal-overlay');
        if (modal) {
            modal.classList.remove('active');
        }

        // Show success notification with name
        const displayName = result.name || result.email;
        Notifications.success(`✅ Logged in as ${displayName}`);

        // Reload status and update UI
        this.loadStatus().then(() => {
            this.updateUI();
        });
    },

    /**
     * Handle failed login (called from backend)
     * @param {Object} result - Error result
     */
    onLoginFailed(result) {
        console.error('[BackupEvents] Login failed:', result);

        // Close loading modal properly
        const modal = document.getElementById('modal-overlay');
        if (modal) {
            modal.classList.remove('active');
        }

        // Show error notification
        Notifications.error(`Login failed: ${result.error || 'Unknown error'}`);
    },

    /**
     * Complete Google login with authorization code (legacy - for manual flow)
     * @param {string} authCode - Authorization code from OAuth
     */
    async completeLogin(authCode) {
        try {
            Notifications.info('Completing login...');

            const result = await API.Backup.completeLogin(authCode);

            if (result.success) {
                Notifications.success(`Logged in as ${result.email}`);
                await this.loadStatus();
                this.updateUI();
            } else {
                Notifications.error(`Login failed: ${result.error}`);
            }
        } catch (error) {
            console.error('[BackupEvents] Failed to complete login:', error);
            Notifications.error('Failed to complete login');
        }
    },

    /**
     * Logout from Google Drive
     */
    async logout() {
        try {
            UI.showModal({
                title: 'Logout from Google Drive',
                body: 'Are you sure you want to logout? Auto backup will be disabled.',
                confirmText: 'Logout',
                onConfirm: async () => {
                    const result = await API.Backup.logout();
                    if (result.success) {
                        Notifications.success('Logged out successfully');
                        await this.loadStatus();
                        this.updateUI();
                    } else {
                        Notifications.error('Failed to logout');
                    }
                }
            });
        } catch (error) {
            console.error('[BackupEvents] Failed to logout:', error);
            Notifications.error('Failed to logout');
        }
    },

    // ==================== Backup/Restore ====================

    /**
     * Backup settings to Google Drive
     */
    async backup() {
        try {
            // Check if logged in
            if (!this.currentStatus || !this.currentStatus.is_logged_in) {
                // Show login modal
                await this.showLoginModal();
                return;
            }

            Notifications.info('Backing up to Google Drive...');

            const result = await API.Backup.backup();

            if (result.success) {
                const message = `Backed up ${result.files_backed_up.length} file(s)`;
                Notifications.success(message);
                await this.loadStatus();
                this.updateUI();
            } else if (result.require_login) {
                await this.showLoginModal();
            } else {
                Notifications.error(`Backup failed: ${result.error}`);
            }
        } catch (error) {
            console.error('[BackupEvents] Failed to backup:', error);
            Notifications.error('Failed to backup');
        }
    },

    /**
     * Restore settings from Google Drive
     */
    async restore() {
        try {
            // Check if logged in
            if (!this.currentStatus || !this.currentStatus.is_logged_in) {
                Notifications.error('Please login first');
                return;
            }

            UI.showModal({
                title: 'Restore from Google Drive',
                body: 'This will replace your current settings with the backup from Google Drive. Continue?',
                confirmText: 'Restore',
                onConfirm: async () => {
                    Notifications.info('Restoring from Google Drive...');

                    const result = await API.Backup.restore();

                    if (result.success) {
                        const message = `Restored ${result.files_restored.length} file(s)`;
                        Notifications.success(message);

                        // Reload settings
                        const settings = await API.loadSettings();
                        AppState.loadFromSettings(settings);

                        // Refresh UI
                        await SoundEvents.refreshSounds();
                        UI.updateStopKeybindUI();
                    } else {
                        Notifications.error(`Restore failed: ${result.error}`);
                    }
                }
            });
        } catch (error) {
            console.error('[BackupEvents] Failed to restore:', error);
            Notifications.error('Failed to restore');
        }
    },

    /**
     * Toggle auto backup
     * @param {boolean} enabled - True to enable, false to disable
     */
    async toggleAutoBackup(enabled) {
        try {
            const result = await API.Backup.setAutoBackup(enabled);

            if (result.success) {
                this.currentStatus.auto_backup_enabled = enabled;
                Notifications.success(`Auto backup ${enabled ? 'enabled' : 'disabled'}`);
            } else {
                Notifications.error('Failed to update auto backup setting');
                // Revert toggle
                const toggle = document.getElementById('auto-backup-toggle');
                if (toggle) toggle.checked = !enabled;
            }
        } catch (error) {
            console.error('[BackupEvents] Failed to toggle auto backup:', error);
            Notifications.error('Failed to update auto backup');
        }
    }
};

// Export to global scope
window.BackupEvents = BackupEvents;
