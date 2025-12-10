/**
 * Effects Events - Event handlers for audio effects controls
 */

const EffectsEvents = {

    currentItemType: null,
    currentIdentifier: null,

    /**
     * Setup all effects event listeners
     * @param {string} itemType - 'sound', 'youtube', or 'tiktok'
     * @param {string} identifier - Sound name or URL
     * @param {Object} currentEffects - Current effects configuration
     */
    setupEffectsListeners(itemType, identifier, currentEffects) {
        this.currentItemType = itemType;
        this.currentIdentifier = identifier;

        // Reverb
        this.setupEffectToggle('reverb', currentEffects);
        this.setupEffectParam('reverb-size', 'reverb', 'roomSize');
        this.setupEffectParam('reverb-damp', 'reverb', 'damping');

        // Echo
        this.setupEffectToggle('echo', currentEffects);
        this.setupEffectParam('echo-delay', 'echo', 'delay');
        this.setupEffectParam('echo-feedback', 'echo', 'feedback');

        // Bass Boost
        this.setupEffectToggle('bass', currentEffects);
        this.setupEffectParam('bass-gain', 'bassBoost', 'gain');

        // High Pass
        this.setupEffectToggle('highpass', currentEffects);
        this.setupEffectParam('highpass-cutoff', 'highpass', 'cutoff');

        // Distortion
        this.setupEffectToggle('distortion', currentEffects);
        this.setupEffectParam('distortion-drive', 'distortion', 'drive');
    },

    /**
     * Setup toggle switch for an effect
     * @param {string} effectId - Effect ID (reverb, echo, etc.)
     * @param {Object} currentEffects - Current effects configuration
     */
    setupEffectToggle(effectId, currentEffects) {
        const toggle = document.getElementById(`effect-${effectId}-toggle`);
        const params = document.getElementById(`${effectId}-params`);

        if (toggle && params) {
            toggle.addEventListener('change', async (e) => {
                const enabled = e.target.checked;

                // Show/hide parameters
                if (enabled) {
                    params.classList.remove('hidden');
                } else {
                    params.classList.add('hidden');
                }

                // Map effectId to actual effect name
                const effectName = effectId === 'bass' ? 'bassBoost' : effectId;

                // Update effect
                await this.updateEffect(effectName, 'enabled', enabled);
            });
        }
    },

    /**
     * Setup parameter slider for an effect
     * @param {string} elementId - Input element ID
     * @param {string} effectName - Effect name in config
     * @param {string} paramName - Parameter name
     */
    setupEffectParam(elementId, effectName, paramName) {
        const input = document.getElementById(elementId);
        const display = document.getElementById(`${elementId}-val`);

        if (input && display) {
            // Update display value on input
            input.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                // Format display based on parameter type
                if (paramName === 'delay') {
                    display.textContent = Math.round(value);
                } else if (paramName === 'gain' || paramName === 'cutoff') {
                    display.textContent = Math.round(value);
                } else {
                    display.textContent = value.toFixed(2);
                }
            });

            // Update backend on change
            input.addEventListener('change', async (e) => {
                const value = parseFloat(e.target.value);
                await this.updateEffect(effectName, paramName, value);
            });
        }
    },

    /**
     * Update effect parameter and send to backend
     * @param {string} effectName - Effect name
     * @param {string} paramName - Parameter name
     * @param {*} value - New value
     */
    async updateEffect(effectName, paramName, value) {
        try {
            // Get current effects from state
            let effects;
            if (this.currentItemType === 'sound') {
                effects = AppState.getSoundEffects(this.currentIdentifier);
            } else if (this.currentItemType === 'youtube') {
                effects = AppState.getYoutubeEffects(this.currentIdentifier);
            } else if (this.currentItemType === 'tiktok') {
                effects = AppState.getTikTokEffects(this.currentIdentifier);
            }

            // Update parameter
            effects[effectName][paramName] = value;

            // Save to state
            if (this.currentItemType === 'sound') {
                AppState.setSoundEffects(this.currentIdentifier, effects);
                await eel.set_sound_effects(effects)();
            } else if (this.currentItemType === 'youtube') {
                AppState.setYoutubeEffects(this.currentIdentifier, effects);
                await eel.set_youtube_effects(effects)();
            } else if (this.currentItemType === 'tiktok') {
                AppState.setTikTokEffects(this.currentIdentifier, effects);
                await eel.set_tiktok_effects(effects)();
            }

            // Save settings with proper state
            await API.saveSettings(AppState.toSettings());

        } catch (error) {
            console.error('Error updating effect:', error);
        }
    }
};
