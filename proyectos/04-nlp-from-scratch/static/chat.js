/**
 * Don Quijote GPT - Chat Frontend
 * Handles API communication, device toggle, and message display
 */

// ============ Configuration ============

const API_BASE = '';  // Same origin

// ============ DOM Elements ============

const elements = {
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    chatForm: document.getElementById('chatForm'),
    sendButton: document.getElementById('sendButton'),
    charCount: document.getElementById('charCount'),
    numGenerate: document.getElementById('numGenerate'),
    temperature: document.getElementById('temperature'),
    tempValue: document.getElementById('tempValue'),
    typingIndicator: document.getElementById('typingIndicator'),
    deviceToggle: document.getElementById('deviceToggle'),
    deviceIcon: document.getElementById('deviceIcon'),
    deviceName: document.getElementById('deviceName'),
    gpuStatus: document.getElementById('gpuStatus'),
    errorToast: document.getElementById('errorToast'),
    toastMessage: document.getElementById('toastMessage')
};

// ============ State ============

let state = {
    isGenerating: false,
    currentDevice: 'cpu',
    gpuAvailable: false,
    gpuName: null
};

// ============ API Functions ============

async function fetchDeviceInfo() {
    try {
        const response = await fetch(`${API_BASE}/api/device`);
        if (!response.ok) throw new Error('Failed to get device info');

        const data = await response.json();
        state.currentDevice = data.device;
        state.gpuAvailable = data.gpu_available;
        state.gpuName = data.gpu_name;

        updateDeviceUI();
    } catch (error) {
        console.error('Error fetching device info:', error);
        showToast('Error al obtener información del dispositivo');
    }
}

async function changeDevice(device) {
    if (state.isGenerating) {
        showToast('Espera a que termine la generación actual');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/device`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to change device');
        }

        const data = await response.json();
        state.currentDevice = data.device;
        updateDeviceUI();

    } catch (error) {
        console.error('Error changing device:', error);
        showToast(error.message);
        // Revert toggle
        elements.deviceToggle.checked = state.currentDevice === 'gpu';
    }
}

async function generateText(prompt, numGenerate, temperature) {
    const response = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            prompt: prompt,
            num_generate: parseInt(numGenerate),
            temperature: parseFloat(temperature)
        })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Error generating text');
    }

    return response.json();
}

// ============ UI Functions ============

function updateDeviceUI() {
    // Update toggle state
    elements.deviceToggle.checked = state.currentDevice === 'gpu';

    // Update device info
    elements.deviceIcon.textContent = state.currentDevice === 'gpu' ? '🎮' : '💻';
    elements.deviceName.textContent = state.currentDevice.toUpperCase();

    // Update GPU status
    const statusEl = elements.gpuStatus;
    const statusText = statusEl.querySelector('.status-text');

    statusEl.classList.remove('available', 'unavailable');

    if (state.gpuAvailable) {
        statusEl.classList.add('available');
        statusText.textContent = state.gpuName || 'GPU Disponible';
    } else {
        statusEl.classList.add('unavailable');
        statusText.textContent = 'GPU No Disponible';
    }

    // Enable/disable toggle based on GPU availability
    elements.deviceToggle.disabled = !state.gpuAvailable && state.currentDevice === 'cpu';
}

function addMessage(type, content, avatar) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const avatarEmoji = avatar || (type === 'user' ? '👤' : '🎭');

    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="message-avatar">${avatarEmoji}</div>
            <div class="message-bubble">
                <p class="message-text">${escapeHtml(content)}</p>
            </div>
        </div>
    `;

    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function showTyping(show) {
    elements.typingIndicator.classList.toggle('active', show);
    if (show) scrollToBottom();
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message) {
    elements.toastMessage.textContent = message;
    elements.errorToast.classList.add('show');

    setTimeout(() => {
        hideToast();
    }, 5000);
}

function hideToast() {
    elements.errorToast.classList.remove('show');
}

// Make hideToast available globally for the onclick handler
window.hideToast = hideToast;

function setGenerating(generating) {
    state.isGenerating = generating;
    elements.sendButton.disabled = generating;
    elements.messageInput.disabled = generating;
    showTyping(generating);
}

// ============ Event Handlers ============

async function handleSubmit(e) {
    e.preventDefault();

    const prompt = elements.messageInput.value.trim();
    if (!prompt || state.isGenerating) return;

    // Clear input
    elements.messageInput.value = '';
    elements.charCount.textContent = '0';

    // Add user message
    addMessage('user', prompt);

    // Get settings
    const numGenerate = elements.numGenerate.value;
    const temperature = elements.temperature.value;

    // Generate
    setGenerating(true);

    try {
        const result = await generateText(prompt, numGenerate, temperature);
        addMessage('ai', result.generated_text);
    } catch (error) {
        console.error('Generation error:', error);
        addMessage('system', `Error: ${error.message}`, '⚠️');
        showToast('Error al generar texto');
    } finally {
        setGenerating(false);
    }
}

function handleInput() {
    const length = elements.messageInput.value.length;
    elements.charCount.textContent = length;

    // Auto-resize textarea
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 150) + 'px';
}

function handleTemperatureChange() {
    elements.tempValue.textContent = elements.temperature.value;
}

function handleDeviceToggle() {
    const device = elements.deviceToggle.checked ? 'gpu' : 'cpu';
    changeDevice(device);
}

function handleKeyDown(e) {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
    }
}

// ============ Initialization ============

function init() {
    // Event listeners
    elements.chatForm.addEventListener('submit', handleSubmit);
    elements.messageInput.addEventListener('input', handleInput);
    elements.messageInput.addEventListener('keydown', handleKeyDown);
    elements.temperature.addEventListener('input', handleTemperatureChange);
    elements.deviceToggle.addEventListener('change', handleDeviceToggle);

    // Initialize temp display
    elements.tempValue.textContent = elements.temperature.value;

    // Fetch initial device info
    fetchDeviceInfo();

    // Focus input
    elements.messageInput.focus();

    console.log('🏰 Don Quijote GPT Chat initialized');
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
