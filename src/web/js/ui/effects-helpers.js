/**
 * Effects Helper Functions
 * Utilities for rendering and managing audio effects UI
 */

const EffectsHelpers = {

    /**
     * Get default effects configuration
     */
    getDefaultEffects() {
        return {
            reverb: { enabled: false, roomSize: 0.5, damping: 0.5 },
            echo: { enabled: false, delay: 250, feedback: 0.3 },
            bassBoost: { enabled: false, gain: 6, frequency: 100 },
            highpass: { enabled: false, cutoff: 80 },
            distortion: { enabled: false, drive: 0.5 }
        };
    },

    /**
     * Render effects section HTML
     * @param {Object} currentEffects - Current effects configuration
     * @returns {string} HTML string for effects section
     */
    renderEffectsSection(currentEffects) {
        const effects = currentEffects || this.getDefaultEffects();

        return `
            <div class="panel-section effects-section">
                <h3>Audio Effects</h3>
                
                <!-- Reverb -->
                <div class="effect-control">
                    <div class="effect-header">
                        <label class="switch">
                            <input type="checkbox" id="effect-reverb-toggle" 
                                   ${effects.reverb.enabled ? 'checked' : ''}>
                            <span class="slider"></span>
                        </label>
                        <span>Reverb</span>
                    </div>
                    <div class="effect-params ${effects.reverb.enabled ? '' : 'hidden'}" id="reverb-params">
                        <label>
                            Room Size: 
                            <span id="reverb-size-val">${effects.reverb.roomSize.toFixed(2)}</span>
                        </label>
                        <input type="range" id="reverb-size" min="0" max="1" step="0.01" 
                               value="${effects.reverb.roomSize}">
                        <label>
                            Damping: 
                            <span id="reverb-damp-val">${effects.reverb.damping.toFixed(2)}</span>
                        </label>
                        <input type="range" id="reverb-damp" min="0" max="1" step="0.01" 
                               value="${effects.reverb.damping}">
                    </div>
                </div>
                
                <!-- Echo -->
                <div class="effect-control">
                    <div class="effect-header">
                        <label class="switch">
                            <input type="checkbox" id="effect-echo-toggle" 
                                   ${effects.echo.enabled ? 'checked' : ''}>
                            <span class="slider"></span>
                        </label>
                        <span>Echo</span>
                    </div>
                    <div class="effect-params ${effects.echo.enabled ? '' : 'hidden'}" id="echo-params">
                        <label>
                            Delay (ms): 
                            <span id="echo-delay-val">${effects.echo.delay}</span>
                        </label>
                        <input type="range" id="echo-delay" min="50" max="1000" step="10" 
                               value="${effects.echo.delay}">
                        <label>
                            Feedback: 
                            <span id="echo-feedback-val">${effects.echo.feedback.toFixed(2)}</span>
                        </label>
                        <input type="range" id="echo-feedback" min="0" max="0.9" step="0.05" 
                               value="${effects.echo.feedback}">
                    </div>
                </div>
                
                <!-- Bass Boost -->
                <div class="effect-control">
                    <div class="effect-header">
                        <label class="switch">
                            <input type="checkbox" id="effect-bass-toggle" 
                                   ${effects.bassBoost.enabled ? 'checked' : ''}>
                            <span class="slider"></span>
                        </label>
                        <span>Bass Boost</span>
                    </div>
                    <div class="effect-params ${effects.bassBoost.enabled ? '' : 'hidden'}" id="bass-params">
                        <label>
                            Gain (dB): 
                            <span id="bass-gain-val">${effects.bassBoost.gain}</span>
                        </label>
                        <input type="range" id="bass-gain" min="0" max="12" step="1" 
                               value="${effects.bassBoost.gain}">
                    </div>
                </div>
                
                <!-- High Pass Filter -->
                <div class="effect-control">
                    <div class="effect-header">
                        <label class="switch">
                            <input type="checkbox" id="effect-highpass-toggle" 
                                   ${effects.highpass.enabled ? 'checked' : ''}>
                            <span class="slider"></span>
                        </label>
                        <span>High Pass</span>
                    </div>
                    <div class="effect-params ${effects.highpass.enabled ? '' : 'hidden'}" id="highpass-params">
                        <label>
                            Cutoff (Hz): 
                            <span id="highpass-cutoff-val">${effects.highpass.cutoff}</span>
                        </label>
                        <input type="range" id="highpass-cutoff" min="20" max="500" step="10" 
                               value="${effects.highpass.cutoff}">
                    </div>
                </div>
                
                <!-- Distortion -->
                <div class="effect-control">
                    <div class="effect-header">
                        <label class="switch">
                            <input type="checkbox" id="effect-distortion-toggle" 
                                   ${effects.distortion.enabled ? 'checked' : ''}>
                            <span class="slider"></span>
                        </label>
                        <span>Distortion</span>
                    </div>
                    <div class="effect-params ${effects.distortion.enabled ? '' : 'hidden'}" id="distortion-params">
                        <label>
                            Drive: 
                            <span id="distortion-drive-val">${effects.distortion.drive.toFixed(2)}</span>
                        </label>
                        <input type="range" id="distortion-drive" min="0" max="1" step="0.05" 
                               value="${effects.distortion.drive}">
                    </div>
                </div>
            </div>
        `;
    }
};
