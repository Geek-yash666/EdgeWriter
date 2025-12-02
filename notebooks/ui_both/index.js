import { FilesetResolver as GenAiFilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai';

// === DOM Elements ===
const input = document.getElementById('input');
const output = document.getElementById('output');
const submit = document.getElementById('submit');
const submitText = document.getElementById('submit-text');
const spinner = document.getElementById('spinner');
const taskSelect = document.getElementById('task');
const toneSelect = document.getElementById('tone');
const customToneInput = document.getElementById('custom-tone-input');
const toneContainer = document.getElementById('tone-container');
const deviceStatus = document.getElementById('device-status');
const statusIndicator = document.getElementById('status-indicator');
const loader = document.getElementById('loader');
const loaderStatus = document.getElementById('loader-status');
const progressBar = document.getElementById('progress-bar');

// Metrics elements
const metricEngine = document.getElementById('metric-engine');
const metricTime = document.getElementById('metric-time');
const metricTokens = document.getElementById('metric-tokens');
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

// Model selection buttons
const modeBaseBtn = document.getElementById('mode-base');
const modePhi3Btn = document.getElementById('mode-phi3');
const baseStatus = document.getElementById('base-status');
const phi3Status = document.getElementById('phi3-status');

// === Config ===
const MEDIAPIPE_MODEL = '../base_model/weights.bin';
const PHI3_SERVER_URL = ''; // Same origin - server.py serves both UI and API

// === State ===
let selectedMode = null;
let mediapipeLLM = null;
let isPhi3ServerReady = false;
let isBaseModelReady = false;
let gpuSupportProbe = null;

// === Hardware Detection ===
async function detectHardware() {
  const userAgent = navigator.userAgent;
  let osName = 'Unknown';
  
  if (userAgent.includes('Windows')) osName = 'Windows';
  else if (userAgent.includes('Mac')) osName = 'macOS';
  else if (userAgent.includes('Linux')) osName = 'Linux';
  else if (userAgent.includes('Android')) osName = 'Android';
  else if (userAgent.includes('iOS') || userAgent.includes('iPhone') || userAgent.includes('iPad')) osName = 'iOS';
  
  document.getElementById('hw-platform').textContent = osName;
  
  const cores = navigator.hardwareConcurrency || 'N/A';
  document.getElementById('hw-cores').textContent = cores !== 'N/A' ? `${cores} cores` : 'N/A';
  
  if (navigator.deviceMemory) {
    document.getElementById('hw-memory').textContent = `${navigator.deviceMemory} GB`;
  } else {
    document.getElementById('hw-memory').textContent = 'N/A';
  }
  
  const gpuSupport = await evaluateGpuDelegateSupport();
  const hwGpu = document.getElementById('hw-gpu');
  if (gpuSupport.ok) {
    hwGpu.textContent = 'WebGPU Available';
    hwGpu.classList.add('text-emerald-400');
  } else {
    hwGpu.textContent = gpuSupport.reason || 'Not Available';
    hwGpu.classList.add('text-amber-400');
  }
}

// Probe WebGPU capabilities
async function evaluateGpuDelegateSupport() {
  if (gpuSupportProbe) return gpuSupportProbe;

  gpuSupportProbe = (async () => {
    if (!('gpu' in navigator)) {
      return { ok: false, reason: 'WebGPU not available' };
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        return { ok: false, reason: 'No WebGPU adapter' };
      }

      const requiredFeatures = ['shader-f16'];
      for (const feature of requiredFeatures) {
        if (!adapter.features || !adapter.features.has(feature)) {
          return { ok: false, reason: `Missing: ${feature}` };
        }
      }

      return { ok: true, limits: adapter.limits || {} };
    } catch (err) {
      return { ok: false, reason: err?.message || 'Probe failed' };
    }
  })();

  return gpuSupportProbe;
}

// === Counter Updates ===
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

// === Progress Updates ===
function updateProgress(percent, status = 'Processing...') {
  progressBar.style.width = `${percent}%`;
  loaderStatus.textContent = status;
}

// === Quality Metrics ===
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
  
  const clarityImprovement = originalDeviation > 0 ? Math.max(0, (originalDeviation - refinedDeviation) / originalDeviation * 100) : 0;
  const clarity = Math.min(100, 70 + clarityImprovement * 0.3);
  
  const originalWords = originalText.trim().split(/\s+/).length;
  const refinedWords = refinedText.trim().split(/\s+/).length;
  const reduction = originalWords > 0 ? ((originalWords - refinedWords) / originalWords) * 100 : 0;
  
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
  clarityBar.className = `h-full transition-all duration-500 ${metrics.clarity >= 80 ? 'bg-emerald-400' : metrics.clarity >= 60 ? 'bg-yellow-400' : 'bg-red-400'}`;
  
  conciseScore.textContent = `${metrics.conciseness}%`;
  conciseBar.style.width = `${metrics.conciseness}%`;
  conciseBar.className = `h-full transition-all duration-500 ${metrics.conciseness >= 80 ? 'bg-blue-400' : metrics.conciseness >= 60 ? 'bg-yellow-400' : 'bg-red-400'}`;
  
  improvementScore.textContent = `${metrics.improvement}%`;
  improvementBar.style.width = `${metrics.improvement}%`;
  improvementBar.className = `h-full transition-all duration-500 ${metrics.improvement >= 80 ? 'bg-purple-400' : metrics.improvement >= 60 ? 'bg-yellow-400' : 'bg-red-400'}`;
}

function resetQualityMetrics() {
  clarityScore.textContent = '--';
  clarityBar.style.width = '0%';
  conciseScore.textContent = '--';
  conciseBar.style.width = '0%';
  improvementScore.textContent = '--';
  improvementBar.style.width = '0%';
}

// === Energy Estimation ===
function estimateEnergy(latencyMs, mode) {
  const cpuPowerW = 0.5;
  const gpuPowerW = 3.0;
  const timeSeconds = latencyMs / 1000;
  
  let energyWs;
  if (mode === 'phi3') {
    energyWs = (cpuPowerW + gpuPowerW) * timeSeconds;
  } else {
    energyWs = cpuPowerW * timeSeconds;
  }
  
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
      input: 'Advances in battery chemistry over the past decade have shifted from incremental improvements to structural innovations. Researchers now prioritize energy-dense solid-state architectures, aiming to reduce flammability while extending cycle life far beyond current lithium-ion norms.',
      output: 'Battery development has moved from small refinements to structural innovations, with solid-state architectures prioritized for higher energy density, lower flammability, and longer life.'
    }
  ],
  'Rewrite': {
    'Neutral': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'The system failed to start because of a memory allocation error.' }
    ],
    'Professional': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'The system encountered a startup failure attributable to a memory allocation error.' }
    ],
    'Friendly': [
      { input: 'The system failed to start due to a memory allocation error.', output: "The system couldn't start up because of a memory allocation error." }
    ],
    'Concise': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'System failed: memory allocation error.' }
    ],
    'Academic': [
      { input: 'The system failed to start due to a memory allocation error.', output: 'The system initialization was unsuccessful due to a memory allocation error.' }
    ]
  },
  'Proofread': [
    { input: 'The system faild to start becuase of a memmory allocation error.', output: 'The system failed to start because of a memory allocation error.' }
  ],
  'Paraphrase': [
    { input: 'The system failed to start due to a memory allocation error.', output: 'A memory allocation issue prevented the system from starting.' }
  ]
};

function getTaskInstruction(task, tone = 'Neutral') {
  switch (task) {
    case 'Summarize': 
      return `TASK: Summarize the text in 2-4 sentences.\nRULES:\n- Cover the main points\n- Do NOT add information not in the original\n- Maintain factual accuracy\nOutput only the summary:`;
    case 'Proofread': 
      return `TASK: Fix grammar, spelling, and punctuation errors.\nRULES:\n- Only fix errors, do NOT rewrite\n- Keep the original wording\n- Preserve sentence structure\nOutput only the corrected text:`;
    case 'Paraphrase': 
      return `TASK: Paraphrase while keeping similar length and meaning.\nRULES:\n- Use different words but keep ALL facts\n- Do NOT add or remove information\n- Keep the same approximate length\nOutput only the paraphrased text:`;
    case 'Rewrite':
    default:
      let toneGuidelines = '';
      switch(tone) {
        case 'Professional':
          toneGuidelines = '\nTONE: Use formal, business-appropriate vocabulary.';
          break;
        case 'Friendly':
          toneGuidelines = '\nTONE: Use conversational, warm language.';
          break;
        case 'Concise':
          toneGuidelines = '\nTONE: Be extremely brief. Remove unnecessary words.';
          break;
        case 'Academic':
          toneGuidelines = '\nTONE: Use scholarly vocabulary and formal structures.';
          break;
        case 'Custom':
          const customStyle = customToneInput.value.trim() || 'unique style';
          toneGuidelines = `\nTONE: <${customStyle}>`;
          break;
        default:
          toneGuidelines = '\nTONE: Maintain neutral, clear language.';
      }
      return `TASK: Rewrite the text for better clarity.${toneGuidelines}\nRULES:\n- Keep EVERY piece of information\n- Do NOT add or remove facts\nOutput only the rewritten text:`;
  }
}

function buildBaseModelPrompt(task, userText, tone = 'Neutral') {
  const instruction = getTaskInstruction(task, tone);
  
  if (task === 'Rewrite' && tone === 'Custom') {
    return `${instruction}\nINPUT: ${userText}\nOUTPUT:`;
  }
  
  let examples;
  if (task === 'Rewrite') {
    examples = fewShotExamples['Rewrite'][tone] || fewShotExamples['Rewrite']['Neutral'];
  } else {
    examples = fewShotExamples[task] || fewShotExamples['Proofread'];
  }
  
  const exampleText = examples.map(p => `${instruction}\nINPUT: ${p.input}\nOUTPUT: ${p.output}`).join('\n\n');
  return `${exampleText}\n\n${instruction}\nINPUT: ${userText}\nOUTPUT:`;
}

// === Model Selection ===
window.selectMode = async (mode) => {
  selectedMode = mode;
  
  // Update UI
  modeBaseBtn.classList.remove('selected');
  modePhi3Btn.classList.remove('selected');
  
  if (mode === 'base') {
    modeBaseBtn.classList.add('selected');
    deviceStatus.textContent = "Base Model selected";
    submitText.textContent = isBaseModelReady ? "Generate" : "Initialize Base Model";
  } else if (mode === 'phi3') {
    modePhi3Btn.classList.add('selected');
    deviceStatus.textContent = "Phi-3 selected";
    submitText.textContent = isPhi3ServerReady ? "Generate" : "Check Server Connection";
  }
  
  submit.disabled = false;
  statusIndicator.className = "status-dot";
};

// === Check Phi-3 Server ===
async function checkPhi3Server() {
  console.log("🔄 Checking Phi-3 server...");
  phi3Status.textContent = "Checking...";
  
  try {
    const res = await fetch(`${PHI3_SERVER_URL}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(3000)
    });
    
    if (res.ok) {
      const data = await res.json();
      console.log("✅ Phi-3 server connected!", data);
      isPhi3ServerReady = true;
      phi3Status.textContent = "Server Online";
      phi3Status.classList.remove('text-red-400');
      phi3Status.classList.add('text-emerald-400');
      return true;
    }
  } catch (e) {
    console.log("⚠️ Phi-3 server not available:", e.message);
  }
  
  isPhi3ServerReady = false;
  phi3Status.textContent = "Server Offline";
  phi3Status.classList.remove('text-emerald-400');
  phi3Status.classList.add('text-red-400');
  return false;
}

// === Initialize Base Model ===
async function initBaseModel() {
  loader.classList.remove('hidden');
  updateProgress(10, "Loading base model...");
  deviceStatus.textContent = "Initializing MediaPipe...";
  statusIndicator.className = "status-dot busy";
  baseStatus.textContent = "Loading...";

  try {
    updateProgress(30, "Downloading model...");
    
    const genaiFileset = await GenAiFilesetResolver.forGenAiTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm"
    );
    
    const gpuSupport = await evaluateGpuDelegateSupport();
    let usedGpu = false;
    
    if (gpuSupport.ok) {
      updateProgress(60, "Trying GPU...");
      try {
        mediapipeLLM = await LlmInference.createFromOptions(genaiFileset, {
          baseOptions: { modelAssetPath: MEDIAPIPE_MODEL, delegate: 'GPU' },
          maxTokens: 4096,
          temperature: 0.5,
          topK: 40
        });
        usedGpu = true;
      } catch (gpuError) {
        console.warn("GPU init failed, using CPU:", gpuError);
      }
    }

    if (!usedGpu) {
      updateProgress(70, "Using CPU...");
      mediapipeLLM = await LlmInference.createFromOptions(genaiFileset, {
        baseOptions: { modelAssetPath: MEDIAPIPE_MODEL, delegate: 'CPU' },
        maxTokens: 4096,
        temperature: 0.5,
        topK: 40
      });
    }

    updateProgress(100, "Ready!");
    isBaseModelReady = true;
    baseStatus.textContent = usedGpu ? "GPU Ready" : "CPU Ready";
    baseStatus.classList.add('text-emerald-400');
    deviceStatus.textContent = usedGpu ? "Base Model Ready (GPU)" : "Base Model Ready (CPU)";
    statusIndicator.className = "status-dot active";
    submitText.textContent = "Generate";
    
    setTimeout(() => { loader.classList.add('hidden'); }, 500);
    return true;

  } catch (e) {
    console.error("Base model init failed:", e);
    loader.classList.add('hidden');
    baseStatus.textContent = "Failed";
    baseStatus.classList.add('text-red-400');
    deviceStatus.textContent = "Initialization failed";
    statusIndicator.className = "status-dot error";
    alert("Failed to load base model. Check console for details.");
    return false;
  }
}

// === Generate with Base Model ===
async function generateWithBaseModel(task, tone, userText) {
  const fullPrompt = buildBaseModelPrompt(task, userText, tone);
  
  let fullResult = "";
  const start = performance.now();
  
  return new Promise((resolve, reject) => {
    mediapipeLLM.generateResponse(fullPrompt, (partialResult, complete) => {
      output.value += partialResult;
      fullResult += partialResult;
      output.scrollTop = output.scrollHeight;

      if (complete) {
        const latency = Math.round(performance.now() - start);
        resolve({ result: fullResult, latency });
      }
    });
  });
}

// === Generate with Phi-3 ===
async function generateWithPhi3(task, tone, userText) {
  const start = performance.now();
  
  const res = await fetch(`${PHI3_SERVER_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      task, 
      tone,
      custom_tone: tone === 'Custom' ? customToneInput.value.trim() : '',
      text: userText 
    })
  });
  
  const data = await res.json();
  const latency = Math.round(performance.now() - start);
  
  return { 
    result: data.text, 
    latency,
    serverLatency: data.latency,
    tokens: data.tokens 
  };
}

// === Main Generate Function ===
async function generate() {
  if (!selectedMode) {
    alert("Please select a model first (Base or Phi-3)");
    return;
  }

  // Initialize if needed
  if (selectedMode === 'base' && !isBaseModelReady) {
    const success = await initBaseModel();
    if (!success) return;
    return;
  }

  if (selectedMode === 'phi3' && !isPhi3ServerReady) {
    loader.classList.remove('hidden');
    updateProgress(50, "Checking server...");
    const connected = await checkPhi3Server();
    loader.classList.add('hidden');
    
    if (!connected) {
      alert("Phi-3 server is not running.\n\nPlease start it with:\n  cd phi_model\n  python server.py");
      return;
    }
    
    deviceStatus.textContent = "Phi-3 Ready";
    statusIndicator.className = "status-dot active";
    submitText.textContent = "Generate";
    return;
  }

  const text = input.value.trim();
  if (!text) {
    alert("Please enter some text to process");
    return;
  }

  const task = taskSelect.value;
  const tone = toneSelect.value;

  // Reset UI
  output.value = "";
  resetQualityMetrics();
  submit.disabled = true;
  submitText.textContent = "Generating...";
  spinner?.classList.remove("hidden");
  loader.classList.remove('hidden');
  updateProgress(30, `Processing with ${selectedMode === 'phi3' ? 'Phi-3' : 'Base Model'}...`);
  
  deviceStatus.textContent = `Generating (${task})...`;
  statusIndicator.className = "status-dot busy";

  // Reset metrics
  metricEngine.textContent = "--";
  metricTime.textContent = "--";
  metricTokens.textContent = "--";
  metricEnergy.textContent = "--";

  try {
    let result, latency, tokens;

    if (selectedMode === 'phi3') {
      const data = await generateWithPhi3(task, tone, text);
      result = data.result;
      latency = data.latency;
      tokens = data.tokens;
      
      output.value = result;
      
      metricEngine.textContent = "Phi-3 Mini";
      metricTime.textContent = `${data.serverLatency}s`;
      metricTokens.textContent = tokens ? `${tokens.prompt}+${tokens.completion}` : "--";
      metricEnergy.textContent = estimateEnergy(latency, 'phi3');
      
    } else {
      const data = await generateWithBaseModel(task, tone, text);
      result = data.result;
      latency = data.latency;
      
      metricEngine.textContent = "Base (MediaPipe)";
      metricTime.textContent = `${latency}ms`;
      metricTokens.textContent = "--";
      metricEnergy.textContent = estimateEnergy(latency, 'base');
    }

    // Calculate quality metrics
    const metrics = calculateQualityMetrics(text, result);
    displayQualityMetrics(metrics);

    deviceStatus.textContent = `Done – ${selectedMode === 'phi3' ? 'Phi-3' : 'Base'} inference`;
    statusIndicator.className = "status-dot active";
    
    updateProgress(100, "Complete!");
    setTimeout(() => { loader.classList.add('hidden'); }, 500);

  } catch (e) {
    console.error("Generation error:", e);
    output.value = `Error: ${e.message}\n\nPlease check the console for details.`;
    deviceStatus.textContent = "Generation error";
    statusIndicator.className = "status-dot error";
    loader.classList.add('hidden');
  } finally {
    submit.disabled = false;
    submitText.textContent = "Generate";
    spinner?.classList.add("hidden");
  }
}

// === Event Listeners ===
submit.addEventListener('click', generate);
input.addEventListener('keydown', e => e.ctrlKey && e.key === 'Enter' && generate());

// Task change - show/hide tone
taskSelect.addEventListener('change', () => {
  if (taskSelect.value === 'Rewrite') {
    toneContainer.style.display = 'block';
  } else {
    toneContainer.style.display = 'none';
  }
});

// Tone change - show/hide custom input
toneSelect.addEventListener('change', () => {
  if (toneSelect.value === 'Custom') {
    customToneInput.classList.remove('hidden');
    customToneInput.focus();
  } else {
    customToneInput.classList.add('hidden');
  }
});

// === Global Functions ===
window.pasteText = async () => {
  try {
    const text = await navigator.clipboard.readText();
    input.value = text;
    updateCounters();
  } catch (e) { 
    alert("Clipboard access denied"); 
  }
};

window.clearAll = () => {
  input.value = "";
  output.value = "";
  updateCounters();
  resetQualityMetrics();
  metricEngine.textContent = "--";
  metricTime.textContent = "--";
  metricTokens.textContent = "--";
  metricEnergy.textContent = "--";
};

window.copyOutput = () => {
  if (!output.value) return;
  navigator.clipboard.writeText(output.value);
};

// === Initialize ===
console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
console.log("⚡ EdgeWriter Dual Engine initializing...");
console.log("📅", new Date().toLocaleString());

// Set initial UI state
if (taskSelect.value !== 'Rewrite') {
  toneContainer.style.display = 'none';
}
updateCounters();
submit.disabled = true;
submitText.textContent = "Select a Model";

// Detect hardware
detectHardware().then(() => {
  console.log("🖥️ Hardware detection complete");
});

// Check Phi-3 server in background
checkPhi3Server().then(ready => {
  if (ready) {
    console.log("✅ Phi-3 server is available");
  } else {
    console.log("⚠️ Phi-3 server not running. Start with: python server.py");
  }
});

console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");