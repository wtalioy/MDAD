/**
 * QuadVox Interactive Audio Demo
 * Main JavaScript file for audio player management and interactions
 */

// ===========================
// Global State
// ===========================
const audioPlayers = {};
let currentlyPlaying = null;

// ===========================
// Audio File Paths (Placeholders)
// ===========================
const audioFiles = {
    hero: 'assets/audio/hero_masterpiece.wav',
    styleScripted: 'assets/audio/style_scripted.wav',
    styleSpontaneous: 'assets/audio/style_spontaneous.wav',
    styleRealworld: 'assets/audio/style_realworld.wav',
    emotionA: 'assets/audio/emotion_real.wav',      // Real emotional speech
    emotionB: 'assets/audio/emotion_fake.wav',      // Fake emotional speech
    noiseClean: 'assets/audio/noise_clean.wav',
    noiseTraffic: 'assets/audio/noise_traffic.wav',
    partial: 'assets/audio/partial_clip.wav'
};

// Blind test configuration (Sample A is real, Sample B is fake)
// To randomize: Change 'realSample' to 'b' and 'fakeSample' to 'a' and swap audio files
const blindTestConfig = {
    realSample: 'a',  // Which sample is real ('a' or 'b')
    fakeSample: 'b'   // Which sample is fake ('a' or 'b')
};

// ===========================
// Initialize on DOM Load
// ===========================
document.addEventListener('DOMContentLoaded', function () {
    initializeWavesurfers();
    initializeTabSwitching();
    initializePlayButtons();
    initializeNoiseToggle();
    initializeTranscriptReveal();
    initializeBlindTest();
});

// ===========================
// Wavesurfer Initialization
// ===========================
function initializeWavesurfers() {
    // Hero Player
    audioPlayers.hero = WaveSurfer.create({
        container: '#waveform-hero',
        waveColor: '#b19cd9',
        progressColor: '#8e44ad',
        cursorColor: '#8e44ad',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.hero.load(audioFiles.hero);

    // Style Tab - Scripted
    audioPlayers.styleScripted = WaveSurfer.create({
        container: '#waveform-style-scripted',
        waveColor: '#95e1d3',
        progressColor: '#27ae60',
        cursorColor: '#27ae60',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.styleScripted.load(audioFiles.styleScripted);

    // Style Tab - Spontaneous
    audioPlayers.styleSpontaneous = WaveSurfer.create({
        container: '#waveform-style-spontaneous',
        waveColor: '#b19cd9',
        progressColor: '#8e44ad',
        cursorColor: '#8e44ad',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.styleSpontaneous.load(audioFiles.styleSpontaneous);

    // Style Tab - Real-World
    audioPlayers.styleRealworld = WaveSurfer.create({
        container: '#waveform-style-realworld',
        waveColor: '#b19cd9',
        progressColor: '#8e44ad',
        cursorColor: '#8e44ad',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.styleRealworld.load(audioFiles.styleRealworld);

    // Emotion Tab - Sample A (starts with neutral gray)
    audioPlayers.emotionA = WaveSurfer.create({
        container: '#waveform-emotion-a',
        waveColor: '#bdc3c7',
        progressColor: '#7f8c8d',
        cursorColor: '#7f8c8d',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.emotionA.load(audioFiles.emotionA);

    // Emotion Tab - Sample B (starts with neutral gray)
    audioPlayers.emotionB = WaveSurfer.create({
        container: '#waveform-emotion-b',
        waveColor: '#bdc3c7',
        progressColor: '#7f8c8d',
        cursorColor: '#7f8c8d',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.emotionB.load(audioFiles.emotionB);

    // Acoustic Tab - Clean
    audioPlayers.noiseClean = WaveSurfer.create({
        container: '#waveform-noise-clean',
        waveColor: '#b19cd9',
        progressColor: '#8e44ad',
        cursorColor: '#8e44ad',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.noiseClean.load(audioFiles.noiseClean);

    // Acoustic Tab - Traffic
    audioPlayers.noiseTraffic = WaveSurfer.create({
        container: '#waveform-noise-traffic',
        waveColor: '#f8b4b4',
        progressColor: '#e74c3c',
        cursorColor: '#e74c3c',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.noiseTraffic.load(audioFiles.noiseTraffic);

    // Manipulation Tab - Partial
    audioPlayers.partial = WaveSurfer.create({
        container: '#waveform-partial',
        waveColor: '#b19cd9',
        progressColor: '#8e44ad',
        cursorColor: '#8e44ad',
        barWidth: 3,
        barRadius: 3,
        height: 80,
        responsive: true,
        normalize: true
    });
    audioPlayers.partial.load(audioFiles.partial);

    // Add finish event listeners to reset play buttons
    Object.keys(audioPlayers).forEach(key => {
        audioPlayers[key].on('finish', function () {
            // Convert camelCase to kebab-case for selector (e.g., 'styleScripted' -> 'style-scripted')
            const kebabCaseKey = key.replace(/([A-Z])/g, '-$1').toLowerCase();
            const btn = document.querySelector(`[data-player="${kebabCaseKey}"]`);
            if (btn) {
                updatePlayButton(btn, false);
            }
            currentlyPlaying = null;
        });
    });
}

// ===========================
// Tab Switching Logic
// ===========================
function initializeTabSwitching() {
    const tabButtons = document.querySelectorAll('.tab-btn');

    tabButtons.forEach(button => {
        button.addEventListener('click', function () {
            const targetTab = this.dataset.tab;

            // Stop all audio
            stopAllAudio();

            // Update active tab button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Update active tab content
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');
        });
    });
}

// ===========================
// Play Button Logic
// ===========================
function initializePlayButtons() {
    const playButtons = document.querySelectorAll('.play-btn');

    playButtons.forEach(button => {
        button.addEventListener('click', function () {
            const playerKey = this.dataset.player;

            // Handle noise player specially (has two tracks)
            if (playerKey === 'noise') {
                handleNoisePlayer(this);
            } else {
                handleStandardPlayer(playerKey, this);
            }
        });
    });
}

function handleStandardPlayer(playerKey, button) {
    // Convert kebab-case to camelCase (e.g., 'style-scripted' -> 'styleScripted')
    const camelCaseKey = playerKey.replace(/-([a-z])/g, (match, letter) => letter.toUpperCase());
    const player = audioPlayers[camelCaseKey];

    if (!player) {
        console.warn(`Player not found for key: ${playerKey} (converted to: ${camelCaseKey})`);
        return;
    }

    if (player.isPlaying()) {
        // Pause current
        player.pause();
        updatePlayButton(button, false);
        currentlyPlaying = null;
    } else {
        // Stop all others
        stopAllAudio();

        // Play this one
        player.play();
        updatePlayButton(button, true);
        currentlyPlaying = camelCaseKey;
    }
}

function handleNoisePlayer(button) {
    const toggle = document.getElementById('noise-toggle');
    const isNoisy = toggle.checked;
    const activePlayer = isNoisy ? audioPlayers.noiseTraffic : audioPlayers.noiseClean;
    const inactivePlayer = isNoisy ? audioPlayers.noiseClean : audioPlayers.noiseTraffic;

    if (activePlayer.isPlaying()) {
        // Pause
        activePlayer.pause();
        updatePlayButton(button, false);
        currentlyPlaying = null;
    } else {
        // Stop all others
        stopAllAudio();

        // Make sure inactive is stopped
        inactivePlayer.pause();

        // Play active
        activePlayer.play();
        updatePlayButton(button, true);
        currentlyPlaying = 'noise';
    }
}

function updatePlayButton(button, isPlaying) {
    const icon = button.querySelector('i');
    if (isPlaying) {
        icon.classList.remove('fa-play');
        icon.classList.add('fa-pause');
        button.classList.add('playing');
    } else {
        icon.classList.remove('fa-pause');
        icon.classList.add('fa-play');
        button.classList.remove('playing');
    }
}

function stopAllAudio() {
    Object.keys(audioPlayers).forEach(key => {
        if (audioPlayers[key].isPlaying()) {
            audioPlayers[key].pause();
        }
    });

    // Reset all play buttons
    const playButtons = document.querySelectorAll('.play-btn');
    playButtons.forEach(btn => updatePlayButton(btn, false));

    currentlyPlaying = null;
}

// ===========================
// Noise Toggle Logic
// ===========================
function initializeNoiseToggle() {
    const toggle = document.getElementById('noise-toggle');
    const cleanWaveform = document.getElementById('waveform-noise-clean');
    const trafficWaveform = document.getElementById('waveform-noise-traffic');
    const noiseLevelBadge = document.getElementById('noise-level-badge');
    const playButton = document.querySelector('[data-player="noise"]');

    if (!toggle) return;

    toggle.addEventListener('change', function () {
        const isNoisy = this.checked;

        // Get current time from whichever is playing
        let currentTime = 0;
        let wasPlaying = false;

        if (audioPlayers.noiseClean.isPlaying()) {
            currentTime = audioPlayers.noiseClean.getCurrentTime();
            wasPlaying = true;
            audioPlayers.noiseClean.pause();
        } else if (audioPlayers.noiseTraffic.isPlaying()) {
            currentTime = audioPlayers.noiseTraffic.getCurrentTime();
            wasPlaying = true;
            audioPlayers.noiseTraffic.pause();
        } else {
            // Get time even if paused
            currentTime = audioPlayers.noiseClean.getCurrentTime() || audioPlayers.noiseTraffic.getCurrentTime() || 0;
        }

        // Switch waveform visibility
        if (isNoisy) {
            cleanWaveform.style.display = 'none';
            trafficWaveform.style.display = 'block';
            noiseLevelBadge.textContent = 'Traffic Noise';
            noiseLevelBadge.classList.remove('badge-gray');
            noiseLevelBadge.classList.add('badge-red');

            // Sync time and resume if was playing
            audioPlayers.noiseTraffic.seekTo(currentTime / audioPlayers.noiseTraffic.getDuration());
            if (wasPlaying) {
                audioPlayers.noiseTraffic.play();
                updatePlayButton(playButton, true);
            }
        } else {
            cleanWaveform.style.display = 'block';
            trafficWaveform.style.display = 'none';
            noiseLevelBadge.textContent = 'Clean';
            noiseLevelBadge.classList.remove('badge-red');
            noiseLevelBadge.classList.add('badge-gray');

            // Sync time and resume if was playing
            audioPlayers.noiseClean.seekTo(currentTime / audioPlayers.noiseClean.getDuration());
            if (wasPlaying) {
                audioPlayers.noiseClean.play();
                updatePlayButton(playButton, true);
            }
        }
    });
}

// ===========================
// Transcript Reveal Logic
// ===========================
function initializeTranscriptReveal() {
    const revealBtn = document.getElementById('reveal-btn');
    const transcript = document.getElementById('transcript');

    if (!revealBtn || !transcript) return;

    let isRevealed = false;

    revealBtn.addEventListener('click', function () {
        const fakeWords = transcript.querySelectorAll('.hidden-fake');

        if (!isRevealed) {
            // Reveal
            fakeWords.forEach(word => {
                word.classList.add('revealed');
            });
            this.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Synthetic Words';
            isRevealed = true;
        } else {
            // Hide
            fakeWords.forEach(word => {
                word.classList.remove('revealed');
            });
            this.innerHTML = '<i class="fas fa-eye"></i> Reveal Synthetic Words';
            isRevealed = false;
        }
    });
}

// ===========================
// Error Handling for Missing Audio Files
// ===========================
Object.keys(audioPlayers).forEach(key => {
    audioPlayers[key].on('error', function (error) {
        console.warn(`Audio file not found for ${key}:`, error);
        // You can add UI feedback here if needed
    });
});

// ===========================
// Blind Test Logic
// ===========================
function initializeBlindTest() {
    const revealBtn = document.getElementById('emotion-reveal-btn');
    const infoNote = document.getElementById('emotion-info-note');

    if (!revealBtn) return;

    let isRevealed = false;

    revealBtn.addEventListener('click', function () {
        if (isRevealed) return;

        isRevealed = true;

        // Update button state
        this.classList.add('revealed');
        this.innerHTML = '<i class="fas fa-check"></i> Revealed';
        this.disabled = true;

        // Get the real and fake sample elements
        const realSample = blindTestConfig.realSample;
        const fakeSample = blindTestConfig.fakeSample;

        const realCard = document.querySelector(`[data-sample="${realSample}"]`);
        const fakeCard = document.querySelector(`[data-sample="${fakeSample}"]`);

        const realLabel = document.querySelector(`[data-label="${realSample}"]`);
        const fakeLabel = document.querySelector(`[data-label="${fakeSample}"]`);

        // Reveal real sample (green)
        if (realLabel) {
            realLabel.classList.remove('label-gray');
            realLabel.classList.add('revealed-real');
            realLabel.querySelector('.label-text').textContent = 'Real Emotional Speech';
            realLabel.querySelector('i').className = 'fas fa-check-circle';
        }

        if (realCard) {
            const title = realCard.querySelector('.blind-title');
            const subtitle = realCard.querySelector('.blind-subtitle');
            const badges = realCard.querySelector('.blind-badges');

            if (title) title.textContent = 'Real Emotional Speech';
            if (subtitle) subtitle.textContent = 'Human Prosody';
            if (badges) {
                badges.innerHTML = '<span class="badge badge-green">Real</span><span class="badge badge-green">Angry</span>';
            }

            // Update waveform color to green
            const player = audioPlayers[`emotion${realSample.toUpperCase()}`];
            if (player) {
                player.setOptions({
                    waveColor: '#95e1d3',
                    progressColor: '#27ae60',
                    cursorColor: '#27ae60'
                });
            }
        }

        // Reveal fake sample (purple)
        if (fakeLabel) {
            fakeLabel.classList.remove('label-gray');
            fakeLabel.classList.add('revealed-fake');
            fakeLabel.querySelector('.label-text').textContent = 'QuadVox Fake Emotional';
            fakeLabel.querySelector('i').className = 'fas fa-exclamation-triangle';
        }

        if (fakeCard) {
            const title = fakeCard.querySelector('.blind-title');
            const subtitle = fakeCard.querySelector('.blind-subtitle');
            const badges = fakeCard.querySelector('.blind-badges');

            if (title) title.textContent = 'QuadVox Fake Emotional';
            if (subtitle) subtitle.textContent = 'Synthetic Prosody';
            if (badges) {
                badges.innerHTML = '<span class="badge badge-purple">Deepfake</span><span class="badge badge-purple">F5TTS</span>';
            }

            // Update waveform color to purple
            const player = audioPlayers[`emotion${fakeSample.toUpperCase()}`];
            if (player) {
                player.setOptions({
                    waveColor: '#b19cd9',
                    progressColor: '#8e44ad',
                    cursorColor: '#8e44ad'
                });
            }
        }

        // Show info note
        if (infoNote) {
            infoNote.style.display = '';
        }
    });
}

// ===========================
// Mobile-friendly adjustments
// ===========================
function handleMobileView() {
    const isMobile = window.innerWidth <= 768;

    if (isMobile) {
        // Adjust waveform heights for mobile
        Object.keys(audioPlayers).forEach(key => {
            if (audioPlayers[key].params) {
                audioPlayers[key].params.height = 60;
            }
        });
    }
}

// Listen for resize events
window.addEventListener('resize', handleMobileView);
handleMobileView();

