// Backup API Client for Google Drive Integration

/**
 * Backup API wrapper
 * Handles all backup-related API calls to backend
 */
const BackupAPI = {
    /**
     * Get current backup status
     * @returns {Promise<Object>} Backup status
     */
    async getStatus() {
        try {
            return await eel.get_backup_status()();
        } catch (error) {
            console.error('[BackupAPI] Failed to get status:', error);
            return {
                is_logged_in: false,
                user_email: null,
                last_backup_time: null,
                auto_backup_enabled: false
            };
        }
    },

    /**
     * Start Google OAuth login flow
     * @returns {Promise<Object>} Contains auth_url or error
     */
    async startLogin() {
        try {
            return await eel.start_google_login()();
        } catch (error) {
            console.error('[BackupAPI] Failed to start login:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Complete Google OAuth login
     * @param {string} authResponse - Authorization response URL
     * @returns {Promise<Object>} Login result
     */
    async completeLogin(authResponse) {
        try {
            return await eel.complete_google_login(authResponse)();
        } catch (error) {
            console.error('[BackupAPI] Failed to complete login:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Check login status (for polling)
     * @returns {Promise<Object|null>} Login result or null if still waiting
     */
    async checkLoginStatus() {
        try {
            return await eel.check_login_status()();
        } catch (error) {
            console.error('[BackupAPI] Failed to check login status:', error);
            return null;
        }
    },

    /**
     * Logout from Google Drive
     * @returns {Promise<Object>} Logout result
     */
    async logout() {
        try {
            return await eel.google_logout()();
        } catch (error) {
            console.error('[BackupAPI] Failed to logout:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Backup settings to Google Drive
     * @returns {Promise<Object>} Backup result
     */
    async backup() {
        try {
            return await eel.backup_to_drive()();
        } catch (error) {
            console.error('[BackupAPI] Failed to backup:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Restore settings from Google Drive
     * @returns {Promise<Object>} Restore result
     */
    async restore() {
        try {
            return await eel.restore_from_drive()();
        } catch (error) {
            console.error('[BackupAPI] Failed to restore:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Enable or disable auto backup
     * @param {boolean} enabled - True to enable, false to disable
     * @returns {Promise<Object>} Result
     */
    async setAutoBackup(enabled) {
        try {
            return await eel.set_auto_backup(enabled)();
        } catch (error) {
            console.error('[BackupAPI] Failed to set auto backup:', error);
            return { success: false, error: error.message };
        }
    }
};

// Add to global API object
if (window.API) {
    window.API.Backup = BackupAPI;
} else {
    window.API = { Backup: BackupAPI };
}
