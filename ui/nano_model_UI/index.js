import { FilesetResolver as GenAiFilesetResolver, LlmInference } from './libs/tasks-genai.js';

const input = document.getElementById('input');
const output = document.getElementById('output');
const submit = document.getElementById('submit');
const submitText = document.getElementById('submit-text');
const taskSelect = document.getElementById('task');
const toneSelect = document.getElementById('tone');
const toneContainer = toneSelect.parentElement;
const deviceStatus = document.getElementById('device-status');
const statusIndicator = document.getElementById('status-indicator');
const loader = document.getElementById('loader');
const loaderStatus = document.getElementById('loader-status');
const progressBar = document.getElementById('progress-bar');

const metricEngine = document.getElementById('metric-engine');
const metricTime = document.getElementById('metric-time');
const metricQuality = document.getElementById('metric-quality');
const metricEnergy = document.getElementById('metric-energy');

// Quality metrics elements
const clarityScore = document.getElementById('clarity-score');
const clarityBar = document.getElementById('clarity-bar');
const conciseScore = document.getElementById('concise-score');
const conciseBar = document.getElementById('concise-bar');
const improvementScore = document.getElementById('improvement-score');
const improvementBar = document.getElementById('improvement-bar');

// Counter elements
const charCount = document.getElementById('char-count');
const wordCount = document.getElementById('word-count');
const tokenEst = document.getElementById('token-est');

const MEDIAPIPE_MODEL = 'weights.bin';

let mediapipeLLM = null;
let isInitialized = false;
let gpuSupportProbe = null;

// Hardware info elements
const hwPlatform = document.getElementById('hw-platform');
const hwCores = document.getElementById('hw-cores');
const hwMemory = document.getElementById('hw-memory');
const hwGpu = document.getElementById('hw-gpu');

// Chat mode elements
const writingModeBtn = document.getElementById('writing-mode-btn');
const chatModeBtn = document.getElementById('chat-mode-btn');
const writingSection = document.getElementById('writing-section');
const chatSection = document.getElementById('chat-section');
const outputSection = document.getElementById('output-section');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatClear = document.getElementById('chat-clear');
const chatMessages = document.getElementById('chat-messages');

let currentMode = 'writing'; // 'writing' or 'chat'
let chatHistory = [];
let isProcessing = false; // Global flag for generation state
let stopRequested = false; // Flag to trigger stop

// Mode switching
function switchMode(mode) {
  currentMode = mode;
  
  if (mode === 'writing') {
    writingModeBtn.className = 'flex-1 px-4 py-2 rounded-xl font-medium transition bg-gradient-to-r from-purple-600 to-pink-600 text-white';
    chatModeBtn.className = 'flex-1 px-4 py-2 rounded-xl font-medium transition bg-white/5 text-slate-300 hover:bg-white/10';
    writingSection.classList.remove('hidden');
    chatSection.classList.add('hidden');
    outputSection.classList.remove('lg:col-span-2');
    outputSection.classList.remove('hidden');
  } else {
    chatModeBtn.className = 'flex-1 px-4 py-2 rounded-xl font-medium transition bg-gradient-to-r from-purple-600 to-pink-600 text-white';
    writingModeBtn.className = 'flex-1 px-4 py-2 rounded-xl font-medium transition bg-white/5 text-slate-300 hover:bg-white/10';
    writingSection.classList.add('hidden');
    chatSection.classList.remove('hidden');
    outputSection.classList.add('lg:col-span-2');
    outputSection.classList.add('hidden');
    
    // Enable chat controls if model is initialized
    if (isInitialized) {
      chatInput.disabled = false;
      chatSend.disabled = false;
      chatClear.disabled = false;
      updateChatSendButtonState(false);
    }
  }
}

writingModeBtn.addEventListener('click', () => switchMode('writing'));
chatModeBtn.addEventListener('click', () => switchMode('chat'));

// Chat functions
function addChatMessage(role, content, isGenerating = false, allowHtml = false) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-bubble ${role}${isGenerating ? ' generating' : ''}`;
  if (allowHtml) {
    messageDiv.innerHTML = content;
  } else {
    messageDiv.textContent = content;
  }
  
  const placeholder = chatMessages.querySelector('.text-center');
  if (placeholder) placeholder.remove();
  
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  
  return messageDiv;
}

function addTypingIndicator() {
  const typingDiv = document.createElement('div');
  typingDiv.className = 'chat-bubble assistant';
  typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
  typingDiv.id = 'typing-indicator';
  chatMessages.appendChild(typingDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
  const typing = document.getElementById('typing-indicator');
  if (typing) typing.remove();
}

// Helper to toggle Chat Send button icon
function updateChatSendButtonState(generating) {
  if (generating) {
    // Stop Icon (Square)
    chatSend.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5"><path stroke-linecap="round" stroke-linejoin="round" d="M5.25 7.5A2.25 2.25 0 017.5 5.25h9a2.25 2.25 0 012.25 2.25v9a2.25 2.25 0 01-2.25 2.25h-9a2.25 2.25 0 01-2.25-2.25v-9z" /></svg>`;
    chatSend.classList.add('bg-red-500', 'hover:bg-red-600');
    chatSend.classList.remove('bg-purple-600', 'hover:bg-purple-700');
  } else {
    // Send Icon (Arrow)
    chatSend.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5"><path stroke-linecap="round" stroke-linejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" /></svg>`;
    chatSend.classList.remove('bg-red-500', 'hover:bg-red-600');
    chatSend.classList.add('bg-purple-600', 'hover:bg-purple-700');
  }
}

function buildChatPrompt(userMessage) {
  let prompt = "System: You are a highly capable, thoughtful, and precise assistant. Your goal is to deeply understand the user's intent, ask clarifying questions when needed, think step-by-step through complex problems, provide clear and accurate answers, and proactively anticipate helpful follow-up information. Always prioritize being truthful, nuanced, insightful, and efficient, tailoring your responses specifically to the user's needs and preferences.\n\n";
  
  // Add recent history (limit to avoid token overflow)
  const recentHistory = chatHistory.slice(-1); // Last 1 messages
  for (const msg of recentHistory) {
    const roleLabel = msg.role === 'user' ? 'User' : 'Model';
    prompt += `${roleLabel}: ${msg.content}\n`;
  }
  
  prompt += `User: ${userMessage}\nModel:`;
  return prompt;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderMarkdownBasic(text) {
  if (!text) return '';

  // Escape HTML to prevent XSS
  let html = escapeHtml(text);

  // Code blocks (Pre-process to avoid matching * or _ inside code)
  const codeBlocks = [];
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const preservedCode = code.replace(/\s+$/, '');
    codeBlocks.push({ lang, code: preservedCode });
    return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
  });

  const inlineCode = [];
  html = html.replace(/`([^`]+)`/g, (_, code) => {
    inlineCode.push(escapeHtml(code));
    return `__INLINE_CODE_${inlineCode.length - 1}__`;
  });

  //Basic Formatting
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>'); // Bold
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>'); // Italic
  
  // Headers
  html = html.replace(/^### (.*$)/gm, '<h3 class="text-lg font-bold mt-2">$1</h3>');
  html = html.replace(/^## (.*$)/gm, '<h2 class="text-xl font-bold mt-3">$1</h2>');
  html = html.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold mt-4">$1</h1>');

  // Lists (Simple)
  html = html.replace(/^\s*[-•*]\s+(.*$)/gm, '<li class="ml-4 list-disc">$1</li>');
  html = html.replace(/((?:<li.*<\/li>\n?)+)/g, '<ul class="my-2">$1</ul>');

  html = html.replace(/__INLINE_CODE_(\d+)__/g, (_, id) => {
    return `<code class="bg-slate-700 px-1 rounded text-sm">${inlineCode[id]}</code>`;
  });

  html = html.replace(/__CODE_BLOCK_(\d+)__/g, (_, id) => {
    const block = codeBlocks[id];
    const langClass = block.lang ? ` class="language-${block.lang}"` : '';
    const safeCode = escapeHtml(block.code);
    return `<pre class="bg-slate-800 p-3 rounded-lg my-2 overflow-x-auto"><code${langClass}>${safeCode}</code></pre>`;
  });

  // Paragraphs
  html = html.replace(/\n\n+/g, '<br><br>');
  html = html.replace(/\n/g, '<br>');

  return html;
}

async function sendChatMessage() {
  // If currently processing, this button acts as a stop button
  if (isProcessing) {
    stopRequested = true;
    return;
  }

  const message = chatInput.value.trim();
  if (!message || !isInitialized) return;
  
  // Remove warning banner if present
  const warningBanner = document.getElementById('chat-warning-banner');
  if (warningBanner) warningBanner.remove();
  
  // Add user message
  addChatMessage('user', message);
  chatHistory.push({ role: 'user', content: message });
  chatInput.value = '';
  
  // Set processing state
  isProcessing = true;
  stopRequested = false;
  updateChatSendButtonState(true);
  
  // Disable input while generating
  chatInput.disabled = true;
  
  addTypingIndicator();
  
  const prompt = buildChatPrompt(message);
  
  // Ensure context doesn't exceed model limits
  try {
    if (mediapipeLLM.sizeInTokens) {
      const tokenCount = mediapipeLLM.sizeInTokens(prompt);
      if (tokenCount > 3500) { 
        removeTypingIndicator();
        addChatMessage('assistant', '⚠️ Conversation history is too long. Please clear history to continue.');
        chatInput.disabled = false;
        isProcessing = false;
        updateChatSendButtonState(false);
        return;
      }
    }
  } catch (e) {
    console.warn("Token counting failed:", e);
  }

  try {
    let assistantResponse = '';
    let responseDiv = null;
    
    mediapipeLLM.generateResponse(prompt, (partialResult, complete) => {
      // Check for stop request
      if (stopRequested) {
        complete = true; // Force completion
      }

      if (!responseDiv) {
        removeTypingIndicator();
        responseDiv = addChatMessage('assistant', escapeHtml(partialResult), true, true);
      } else {
        responseDiv.innerHTML += escapeHtml(partialResult);
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }
      
      assistantResponse += partialResult;
      
      if (complete || stopRequested) {
        responseDiv.classList.remove('generating');
        responseDiv.innerHTML = renderMarkdownBasic(assistantResponse);
        chatHistory.push({ role: 'assistant', content: assistantResponse });
        
        // Re-enable input
        chatInput.disabled = false;
        chatInput.focus();
        
        // Reset state
        isProcessing = false;
        stopRequested = false;
        updateChatSendButtonState(false);
      }
      
      // Return true to continue, false to stop
      return !stopRequested;
    });
  } catch (e) {
    removeTypingIndicator();
    addChatMessage('assistant', '❌ Error generating response. Please try again.');
    console.error('Chat error:', e);
    chatInput.disabled = false;
    isProcessing = false;
    updateChatSendButtonState(false);
  }
}

chatSend.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

chatClear.addEventListener('click', () => {
  if (confirm('Clear all chat history?')) {
    chatHistory = [];
    chatMessages.innerHTML = '<div class="text-center text-slate-500 text-sm py-8">Start a conversation...</div>';
  }
});


// Minimal hook: try to apply the force_high_performance_gpu switch when running in Electron.
function applyForceHighPerformanceFlag() {
  try {
    const isElectron = !!(window?.process?.versions?.electron);
    if (isElectron && window.require) {
      const electron = window.require('electron');
      const app = electron?.app || electron?.remote?.app;
      if (app?.commandLine?.appendSwitch) {
        app.commandLine.appendSwitch('force-high-performance-gpu');
        app.commandLine.appendSwitch('disable-gpu-process-crash-limit');
        console.log('[GPU] force-high-performance-gpu applied (Electron).');
        return;
      }
    }
  } catch (err) {
    console.warn('[GPU] Applying force_high_performance_gpu failed:', err);
  }
  console.info('[GPU] Browser detected. To force dGPU, launch the browser with --force-high-performance-gpu or set OS graphics preference to High Performance.');
}

async function detectHardware() {
  // Platform detection
  const platform = navigator.platform || 'Unknown';
  const userAgent = navigator.userAgent;
  let osName = 'Unknown';
  
  if (userAgent.includes('Windows')) osName = 'Windows';
  else if (userAgent.includes('Mac')) osName = 'macOS';
  else if (userAgent.includes('Linux')) osName = 'Linux';
  else if (userAgent.includes('Android')) osName = 'Android';
  else if (userAgent.includes('iOS') || userAgent.includes('iPhone') || userAgent.includes('iPad')) osName = 'iOS';
  
  hwPlatform.textContent = osName;
  
  // CPU cores
  const cores = navigator.hardwareConcurrency || 'N/A';
  hwCores.textContent = cores !== 'N/A' ? `${cores} cores` : 'N/A';
  
  // Memory
  let reportedRam = null;
  if (navigator.deviceMemory) {
    reportedRam = navigator.deviceMemory;
  }

  let serverGpuName = null;
  let serverRamGB = null;
  let serverGpuMemory = null;
  let serverGpus = [];
  let hasDedicatedGpu = false;
  
  try {
    const response = await fetch('/api/gpu-info');
    if (response.ok) {
      const data = await response.json();
      if (data.gpus && data.gpus.length > 0) {
        serverGpus = data.gpus;
        serverGpuName = data.gpus[0].name;
        serverGpuMemory = data.gpus[0].memory;
        serverRamGB = data.ramGB || serverRamGB;
        console.log('Server detected GPUs:', data.gpus);
        
        // Check if system has NVIDIA or AMD GPU
        hasDedicatedGpu = data.gpus.some(gpu => 
          gpu.type === 'NVIDIA' || 
          gpu.name.toLowerCase().includes('nvidia') ||
          gpu.name.toLowerCase().includes('geforce') ||
          gpu.name.toLowerCase().includes('rtx') ||
          gpu.name.toLowerCase().includes('gtx') ||
          (gpu.name.toLowerCase().includes('amd') && !gpu.name.toLowerCase().includes('radeon(tm) graphics')) ||
          gpu.name.toLowerCase().includes('radeon rx') ||
          gpu.name.toLowerCase().includes('radeon pro')
        );
      }
    }
  } catch (e) {
    console.log('Server GPU detection not available:', e.message);
  }
  
  // GPU support check
  const gpuSupport = await evaluateGpuDelegateSupport();

  // Update RAM display
  if (typeof serverRamGB === 'number' && !Number.isNaN(serverRamGB)) {
    hwMemory.textContent = `${serverRamGB} GB`;
    hwMemory.classList.add('text-emerald-400');
  } else if (reportedRam) {
    hwMemory.textContent = `${reportedRam} GB`;
    hwMemory.classList.add('text-slate-200');
  } else {
    hwMemory.textContent = 'N/A';
    hwMemory.classList.add('text-slate-500');
  }
  if (gpuSupport.ok) {
    let gpuName = serverGpuName;
    if (!gpuName && gpuSupport.adapterInfo) {
      gpuName = gpuSupport.adapterInfo.device || gpuSupport.adapterInfo.description || gpuSupport.adapterInfo.vendor || null;
    }
    if (!gpuName) {
      gpuName = 'WebGPU Available';
    }
    
    // Display GPU name (with memory inline when available)
    if (serverGpuMemory && serverGpuMemory !== 'Unknown' && hasDedicatedGpu) {
      hwGpu.textContent = `${gpuName} (${serverGpuMemory})`;
    } else {
      hwGpu.textContent = gpuName;
    }
    hwGpu.classList.add('text-emerald-400');

    // Only show banner if Intel GPU is active BUT dedicated GPU exists in system
    if (gpuName.toLowerCase().includes('intel') && hasDedicatedGpu) {
      hwGpu.classList.remove('text-emerald-400');
      hwGpu.classList.add('text-yellow-400');
      hwGpu.title = "Integrated GPU detected but you have a dedicated GPU available. For better performance, configure your browser to use the High Performance GPU.";
      
      // Show GPU guidance banner only when dedicated GPU is available but not being used
      showGpuGuidanceBanner(gpuName, osName, serverGpus);
    } else if (gpuName.toLowerCase().includes('intel')) {
      // Intel GPU but no dedicated GPU found - just mark it yellow but no banner
      hwGpu.classList.remove('text-emerald-400');
      hwGpu.classList.add('text-yellow-400');
      hwGpu.title = "Integrated GPU detected.";
    }
  } else {
    hwGpu.textContent = gpuSupport.reason || 'Not Available';
    hwGpu.classList.add('text-amber-400');
  }
}

function showGpuGuidanceBanner(gpuName, osName, availableGpus = []) {
  // Check if user has dismissed this banner before
  if (localStorage.getItem('edgewriter-gpu-banner-dismissed') === 'true') {
    return;
  }
  
  const banner = document.getElementById('gpu-guidance-banner');
  const currentGpuName = document.getElementById('gpu-current-name');
  const guidanceMessage = document.getElementById('gpu-guidance-message');
  
  if (!banner) return;
  
  // Update banner content based on detected GPU
  currentGpuName.textContent = gpuName;
  
  // Find dedicated GPU names to show in message
  const dedicatedGpus = availableGpus.filter(gpu => 
    gpu.type === 'NVIDIA' || 
    gpu.name.toLowerCase().includes('nvidia') ||
    gpu.name.toLowerCase().includes('geforce') ||
    gpu.name.toLowerCase().includes('rtx') ||
    gpu.name.toLowerCase().includes('gtx') ||
    (gpu.name.toLowerCase().includes('amd') && !gpu.name.toLowerCase().includes('radeon(tm) graphics')) ||
    gpu.name.toLowerCase().includes('radeon rx') ||
    gpu.name.toLowerCase().includes('radeon pro')
  );
  
  const dedicatedGpuNames = dedicatedGpus.map(g => g.name).join(', ');
  
  // Customize message based on OS
  if (osName === 'Windows') {
    if (dedicatedGpuNames) {
      guidanceMessage.textContent = `Your system has ${dedicatedGpuNames} but it's not being used. Here's how to enable it:`;
    } else {
      guidanceMessage.textContent = 'Your system has a dedicated GPU (NVIDIA/AMD) that isn\'t being used. Here\'s how to enable it:';
    }
  } else if (osName === 'macOS') {
    guidanceMessage.textContent = 'macOS manages GPU selection automatically, but you can try the browser flag option below:';
    // Hide Windows-specific options
    const detailsElements = banner.querySelectorAll('details');
    if (detailsElements[0]) detailsElements[0].style.display = 'none'; // Windows Settings
    if (detailsElements[2]) detailsElements[2].style.display = 'none'; // NVIDIA/AMD Control Panel
  } else if (osName === 'Linux') {
    guidanceMessage.textContent = 'You may be able to force high-performance GPU usage. Try the browser flag option below:';
    // Hide Windows-specific options
    const detailsElements = banner.querySelectorAll('details');
    if (detailsElements[0]) detailsElements[0].style.display = 'none'; // Windows Settings
    if (detailsElements[2]) detailsElements[2].style.display = 'none'; // NVIDIA/AMD Control Panel
  }
  
  // Show the banner
  banner.classList.remove('hidden');
}

async function evaluateGpuDelegateSupport() {
  if (gpuSupportProbe) return gpuSupportProbe;

  gpuSupportProbe = (async () => {
    if (!('gpu' in navigator)) {
      return { ok: false, reason: 'WebGPU not available in this environment' };
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        return { ok: false, reason: 'No compatible WebGPU adapter found' };
      }

      let adapterInfo = {};
      if (adapter.requestAdapterInfo) {
        adapterInfo = await adapter.requestAdapterInfo();
      }

      const requiredFeatures = ['shader-f16'];
      for (const feature of requiredFeatures) {
        if (!adapter.features || !adapter.features.has(feature)) {
          return { ok: false, reason: `Missing WebGPU feature: ${feature}` };
        }
      }

      return { ok: true, limits: adapter.limits || {}, adapterInfo };
    } catch (err) {
      console.warn('WebGPU capability probe failed:', err);
      return { ok: false, reason: err?.message || 'WebGPU capability probe failed' };
    }
  })();

  return gpuSupportProbe;
}

function setDeviceStatus(message, busy = false) {
  if (!deviceStatus) return;
  deviceStatus.textContent = message;
  if (busy) {
    deviceStatus.dataset.busy = 'true';
  } else {
    delete deviceStatus.dataset.busy;
  }
}

function updateCounters() {
  const text = input.value;
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).filter(w => w.length > 0).length : 0;
  
  const estimatedTokens = Math.ceil(words * 1.35); 
  
  charCount.textContent = `${chars} characters`;
  wordCount.textContent = `${words} words`;
  tokenEst.textContent = `~${estimatedTokens} tokens`;
  
  if (estimatedTokens > 4000) {
    tokenEst.classList.add('text-red-400');
    tokenEst.classList.remove('text-slate-400');
  } else {
    tokenEst.classList.remove('text-red-400');
    tokenEst.classList.add('text-slate-400');
  }
}

input.addEventListener('input', updateCounters);

function updateProgress(percent, status = 'Processing...') {
  progressBar.style.width = `${percent}%`;
  loaderStatus.textContent = status;
}

function calculateQualityMetrics(originalText, refinedText) {
  const getAvgSentenceLength = (text) => {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const words = text.trim().split(/\s+/).length;
    return sentences.length > 0 ? words / sentences.length : 0;
  };
  
  const originalAvg = getAvgSentenceLength(originalText);
  const refinedAvg = getAvgSentenceLength(refinedText);
  
  const idealLength = 17.5;
  const originalDeviation = Math.abs(originalAvg - idealLength);
  const refinedDeviation = Math.abs(refinedAvg - idealLength);
  
  const clarityImprovement = Math.max(0, (originalDeviation - refinedDeviation) / originalDeviation * 100);
  const clarity = Math.min(100, 70 + clarityImprovement * 0.3);
  
  const originalWords = originalText.trim().split(/\s+/).length;
  const refinedWords = refinedText.trim().split(/\s+/).length;
  const reduction = ((originalWords - refinedWords) / originalWords) * 100;
  
  let conciseness;
  if (reduction < 0) {
    conciseness = Math.max(50, 80 + reduction);
  } else if (reduction > 30) {
    conciseness = Math.max(60, 100 - (reduction - 30) * 2);
  } else {
    conciseness = Math.min(100, 80 + reduction * 1.5);
  }
  
  const improvement = (clarity * 0.5 + conciseness * 0.5);
  
  return {
    clarity: Math.round(clarity),
    conciseness: Math.round(conciseness),
    improvement: Math.round(improvement)
  };
}

function displayQualityMetrics(metrics) {
  clarityScore.textContent = `${metrics.clarity}%`;
  clarityBar.style.width = `${metrics.clarity}%`;
  clarityBar.className = `h-full transition-all ${metrics.clarity >= 80 ? 'bg-emerald-400' : metrics.clarity >= 60 ? 'bg-yellow-400' : 'bg-red-400'}`;
  
  conciseScore.textContent = `${metrics.conciseness}%`;
  conciseBar.style.width = `${metrics.conciseness}%`;
  conciseBar.className = `h-full transition-all ${metrics.conciseness >= 80 ? 'bg-blue-400' : metrics.conciseness >= 60 ? 'bg-yellow-400' : 'bg-red-400'}`;
  
  improvementScore.textContent = `${metrics.improvement}%`;
  improvementBar.style.width = `${metrics.improvement}%`;
  improvementBar.className = `h-full transition-all ${metrics.improvement >= 80 ? 'bg-purple-400' : metrics.improvement >= 60 ? 'bg-yellow-400' : 'bg-red-400'}`;
}

function resetQualityMetrics() {
  clarityScore.textContent = '--';
  clarityBar.style.width = '0%';
  conciseScore.textContent = '--';
  conciseBar.style.width = '0%';
  improvementScore.textContent = '--';
  improvementBar.style.width = '0%';
}

function estimateEnergy(latencyMs) {
  const cpuPowerW = 0.5;
  const timeSeconds = latencyMs / 1000;
  const energyWs = cpuPowerW * timeSeconds;
  const energyMWh = (energyWs / 3600) * 1000;
  
  if (energyMWh < 0.001) {
    return `${(energyMWh * 1000).toFixed(2)} µWh`;
  } else if (energyMWh < 1) {
    return `${energyMWh.toFixed(3)} mWh`;
  } else if (energyMWh < 1000) {
    return `${energyMWh.toFixed(2)} mWh`;
  } else {
    return `${(energyMWh / 1000).toFixed(3)} Wh`;
  }
}

const fewShotExamples = {
  'Summarize': [
    {
      input: 'Advances in battery chemistry over the past decade have shifted from incremental improvements to structural innovations. Researchers now prioritize energy-dense solid-state architectures, aiming to reduce flammability while extending cycle life far beyond current lithium-ion norms. Supply-chain constraints still impede large-scale deployment, particularly in the sourcing of high-purity lithium and rare-earth stabilizers. Analysts project that firms capable of vertically integrating extraction, refinement, and cell fabrication will dominate market share as grid-scale storage and electric mobility converge.',
      output: 'Battery development has moved from small refinements to structural innovations, with solid-state architectures prioritized for higher energy density, lower flammability, and longer life. Deployment remains limited by supply-chain constraints, and vertically integrated firms are positioned to lead as storage and mobility markets merge.'
    },
    {
      input: 'Large data-center operators face rising pressure to reduce water consumption as global drought conditions intensify. Traditional evaporative cooling systems offer high thermal efficiency but rely on substantial freshwater withdrawals. Emerging alternatives—including liquid immersion cooling and advanced refrigerant-based systems—cut water use dramatically but require significant retrofits and carry higher upfront costs. Regulators are beginning to develop reporting standards that will force operators to disclose real water-use metrics rather than relying on modeled estimates.',
      output: 'Data-center operators face pressure to cut water use as drought conditions worsen. Evaporative cooling is efficient but consumes large volumes of freshwater. New options such as immersion and refrigerant-based cooling reduce water use but demand costly retrofits. Regulators are creating standards that require disclosure of actual water-use metrics.'
    }
  ],
  'Rewrite': {
    'Neutral': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'The system failed to start because of a memory allocation error.' },
      { input: 'Calibration completed; sensors returned stable readings.', output: 'Calibration completed, and the sensors returned stable readings.' }
    ],
    'Professional': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'The system encountered a startup failure attributable to a memory allocation error.' },
      { input: 'Calibration completed; sensors returned stable readings.', output: 'Calibration procedures have been completed successfully; sensor readings remain stable.' }
    ],
    'Friendly': [
      { input: 'The system failed to start due to a memory allocation error.', output: "The system couldn't start up because of a memory allocation error." },
      { input: 'Calibration completed; sensors returned stable readings.', output: 'Great news! Calibration is done and the sensors are giving us stable readings.' }
    ],
    'Concise': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'System failed: memory allocation error.' },
      { input: 'Calibration completed; sensors returned stable readings.', output: 'Calibration done. Sensors stable.' }
    ],
    'Academic': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'The system initialization was unsuccessful due to a memory allocation error.' },
      { input: 'Calibration completed; sensors returned stable readings.', output: 'Calibration procedures were completed; sensors yielded stable readings.' }
    ]
  },
  'default': [
    { input: 'The system failed to start due to a memory allocation error.', output: 'The system failed to start because of a memory allocation error.' },
    { input: 'Calibration completed; sensors returned stable readings.', output: 'Calibration completed, and the sensors returned stable readings.' }
  ]
};

function getTaskInstruction(task, tone = 'Neutral') {
  switch (task) {
    case 'Summarize': 
      return `TASK: Summarize the text in 2-4 sentences, capturing the main progression of ideas.\nRULES:\n- Cover the beginning, middle, and end of the argument\n- Combine related points for conciseness\n- Do NOT add information not in the original\n- Maintain factual accuracy\nOutput only the summary:`;
    case 'Proofread': 
      return `TASK: Fix grammar, spelling, and punctuation errors.\nRULES:\n- Only fix errors, do NOT rewrite or paraphrase\n- Keep the original wording and style\n- Do NOT change facts or meaning\n- Preserve the sentence structure\nOutput only the corrected text:`;
    case 'Paraphrase': 
      return `TASK: Paraphrase while keeping similar length and one to one orderly meaning.\nRULES:\n- Use different words but keep ALL facts\n- Do NOT add or remove information\n- Maintain the same level of detail and structure of sentence\n- Keep the same approximate length\nOutput only the paraphrased text do not change sentence strucute as it is mandatory for paraphrase:`;
    case 'Rewrite':
    default:
      let toneGuidelines = '';
      switch(tone) {
        case 'Professional':
          toneGuidelines = '\nTONE: Use formal, business-appropriate vocabulary. Use complete sentences and precise terminology.';
          break;
        case 'Friendly':
          toneGuidelines = '\nTONE: Use conversational, warm language. Use contractions and relatable phrasing.';
          break;
        case 'Concise':
          toneGuidelines = '\nTONE: Be extremely brief. Remove unnecessary words while keeping all facts.';
          break;
        case 'Academic':
          toneGuidelines = '\nTONE: Use scholarly vocabulary. Use formal academic sentence structures and precise terminology.';
          break;
        case 'Neutral':
        default:
          toneGuidelines = '\nTONE: Maintain neutral, clear language without strong stylistic choices.';
      }
      return `TASK: Rewrite the text for better clarity and readability.${toneGuidelines}\nSTRICT RULES:\n- Keep EVERY piece of information from the original\n- Do NOT add interpretations, explanations, or new facts\n- Do NOT remove ANY details (numbers, qualifiers, specifics)\n- Do NOT change the meaning or implications\n- Preserve all original words when possible, only change structure\nOutput only the rewritten text:`;
  }
}

function buildFullPrompt(task, userText, tone = 'Neutral') {
  const instruction = getTaskInstruction(task, tone);
  
  let examples;
  if (task === 'Rewrite') {
    if (fewShotExamples['Rewrite'][tone]) {
      examples = fewShotExamples['Rewrite'][tone];
    } else {
      examples = fewShotExamples['default'];
    }
  } else if (fewShotExamples[task]) {
    examples = fewShotExamples[task];
  } else {
    examples = fewShotExamples['default'];
  }
  
  const exampleText = examples.map(p => `${instruction}\nINPUT: ${p.input}\nOUTPUT: ${p.output}`).join('\n\n');
  return `${exampleText}\n\n${instruction}\nINPUT: ${userText}\nOUTPUT:`;
}

taskSelect.addEventListener('change', () => {
  if (taskSelect.value === 'Rewrite') {
    toneContainer.style.display = 'block';
  } else {
    toneContainer.style.display = 'none';
  }
});

async function init() {
  loader.classList.remove('hidden');
  submit.disabled = true;
  submitText.textContent = "Loading Model...";
  setDeviceStatus("Initializing...", true);
  statusIndicator.className = "status-dot busy";

  let modelBlobUrl = null;

  try {
    updateProgress(10, "Downloading model...");
    setDeviceStatus("Downloading model...", true);

    // Fetch model once to avoid double download and check size
    let modelPath = MEDIAPIPE_MODEL;
    let modelSize = 0;
    try {
      const response = await fetch(MEDIAPIPE_MODEL);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      modelSize = blob.size;
      modelBlobUrl = URL.createObjectURL(blob);
      modelPath = modelBlobUrl;
      updateProgress(30, "Model downloaded");
    } catch (e) {
      console.warn("Pre-fetch failed, using direct path:", e);
    }
    
    const genaiFileset = await GenAiFilesetResolver.forGenAiTasks(
      "./libs/wasm"
    );
    
    const gpuSupport = await evaluateGpuDelegateSupport();
    if (!gpuSupport.ok && gpuSupport.reason) {
      console.warn(`Skipping GPU delegate: ${gpuSupport.reason}`);
    }

    let usedGpu = false;
    let finalStatus = '';

    // Check if model fits in GPU buffer
    let canUseGpu = gpuSupport.ok;
    if (canUseGpu && modelSize > 0 && gpuSupport.limits) {
        const maxBuffer = gpuSupport.limits.maxBufferSize || 0;
        if (modelSize > maxBuffer) {
            console.warn(`Model size (${(modelSize / (1024 * 1024)).toFixed(1)} MB) exceeds GPU maxBufferSize (${(maxBuffer / (1024 * 1024)).toFixed(1)} MB). Disabling GPU.`);
            canUseGpu = false;
            if (!gpuSupport.reason) gpuSupport.reason = "Model too large for GPU buffer";
        }
    }

    if (canUseGpu) {
      updateProgress(60, "Initializing model on GPU...");
      try {
        mediapipeLLM = await LlmInference.createFromOptions(genaiFileset, {
          baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
          maxTokens: 4096,
          temperature: 0.5,
          topK: 40
        });
        usedGpu = true;
        finalStatus = "Ready – GPU acceleration active";
      } catch (gpuError) {
        console.warn("GPU init failed, falling back to CPU:", gpuError);
      }
    }

    if (!usedGpu) {
      const fallbackStatus = gpuSupport.ok ? "GPU unavailable, switching to CPU…" : "GPU disabled, using CPU…";
      updateProgress(70, fallbackStatus);
      setDeviceStatus(fallbackStatus, true);
      if (!gpuSupport.ok && gpuSupport.reason) {
        setDeviceStatus(`GPU disabled: ${gpuSupport.reason}`, true);
      }
      mediapipeLLM = await LlmInference.createFromOptions(genaiFileset, {
        baseOptions: { modelAssetPath: modelPath, delegate: 'CPU' },
        maxTokens: 4096,
        temperature: 0.5,
        topK: 40
      });
      finalStatus = gpuSupport.ok ? "Ready – CPU fallback" : `Ready – CPU fallback${gpuSupport.reason ? ` (${gpuSupport.reason})` : ''}`;
    }

    updateProgress(100, "Ready!");
    setDeviceStatus(finalStatus || "Ready");
    statusIndicator.className = "status-dot active";
    
    // Cleanup blob URL
    if (modelBlobUrl) {
        URL.revokeObjectURL(modelBlobUrl);
    }

    isInitialized = true;
    submit.disabled = false;
    submitText.textContent = "Generate";
    
    // Enable chat mode controls if in chat mode
    if (currentMode === 'chat') {
      chatInput.disabled = false;
      chatSend.disabled = false;
      chatClear.disabled = false;
      updateChatSendButtonState(false);
    }
    
    setTimeout(() => { loader.classList.add('hidden'); }, 500);
    return true;

  } catch (e) {
    if (modelBlobUrl) URL.revokeObjectURL(modelBlobUrl);
    console.error("Initialization failed:", e);
    loader.classList.add('hidden');
    updateProgress(0, "Initialization failed");
    setDeviceStatus("Initialization failed");
    statusIndicator.className = "status-dot error";
    alert("Failed to initialize model. Check console for details.");
    submit.disabled = false;
    submitText.textContent = "Retry Initialization";
    return false;
  }
}

submit.onclick = async () => {
  if (!isInitialized) {
    const success = await init();
    if (!success) return;
    if (!mediapipeLLM) return;
    return;
  }

  // STOP LOGIC FOR WRITING MODE
  if (isProcessing) {
    stopRequested = true;
    submitText.textContent = "Stopping...";
    return;
  }

  if (!input.value.trim()) {
    alert("Please enter some text to process");
    return;
  }

  const task = taskSelect.value;
  const tone = toneSelect.value;
  const userText = input.value.trim();
  const fullPrompt = buildFullPrompt(task, userText, tone);

  try {
    if (mediapipeLLM.sizeInTokens) {
      const tokenCount = mediapipeLLM.sizeInTokens(fullPrompt);
      console.log("Token count:", tokenCount);
      if (tokenCount > 4000) {
        alert(`Input is too long! (${tokenCount} tokens). The maximum allowed is 4000 tokens. Please reduce the text length.`);
        return;
      }
    }
  } catch (e) {
    console.warn("Token counting failed:", e);
  }

  output.value = "";
  resetQualityMetrics();
  
  // UI State: Generating
  isProcessing = true;
  stopRequested = false;
  submit.disabled = false; // Keep enabled so user can click Stop
  submitText.textContent = "Stop"; // Change text to Stop
  submit.classList.add('bg-red-600', 'hover:bg-red-700');
  submit.classList.remove('bg-purple-600', 'hover:bg-purple-700');

  metricEngine.textContent = "--";
  metricTime.textContent = "--";
  metricQuality.textContent = "--";
  metricEnergy.textContent = "--";

  const start = performance.now();

  try {
    const taskLabel = task === 'Rewrite' ? `${task} (${tone})` : task;
    setDeviceStatus(`Base model generating (${taskLabel})...`, true);
    
    let fullResult = "";

    mediapipeLLM.generateResponse(fullPrompt, (partialResult, complete) => {
      // STOP LOGIC
      if (stopRequested) {
        complete = true;
      }

      if (!stopRequested) {
        output.value += partialResult;
        fullResult += partialResult;
        output.scrollTop = output.scrollHeight;
      }

      if (complete || stopRequested) {
        const latency = Math.round(performance.now() - start);

        metricEngine.textContent = "Base (Nano)";
        metricTime.textContent = `${latency}ms`;
        metricQuality.textContent = "Reliable";
        metricEnergy.textContent = estimateEnergy(latency);
        setDeviceStatus(stopRequested ? "Stopped by user" : "Done – Base inference");

        const metrics = calculateQualityMetrics(userText, fullResult);
        displayQualityMetrics(metrics);
        
        // Reset UI State
        isProcessing = false;
        stopRequested = false;
        submit.disabled = false;
        submitText.textContent = "Generate";
        submit.classList.remove('bg-red-600', 'hover:bg-red-700');
        submit.classList.add('bg-purple-600', 'hover:bg-purple-700');
      }
    });

  } catch (e) {
    console.error("Generation error:", e);
    setDeviceStatus("Generation error - check console");
    statusIndicator.className = "status-dot error";
    
    // Reset UI on error
    isProcessing = false;
    submit.disabled = false;
    submitText.textContent = "Generate";
    submit.classList.remove('bg-red-600', 'hover:bg-red-700');
    submit.classList.add('bg-purple-600', 'hover:bg-purple-700');
  }
};


if (taskSelect.value !== 'Rewrite') {
  toneContainer.style.display = 'none';
}
updateCounters();
submit.disabled = false;
applyForceHighPerformanceFlag();
detectHardware();