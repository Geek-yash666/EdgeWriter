import { FilesetResolver as GenAiFilesetResolver, LlmInference } from './libs/tasks-genai.js';

const input = document.getElementById('input');
const output = document.getElementById('output');
const submit = document.getElementById('submit');
const submitText = document.getElementById('submit-text');
const taskSelect = document.getElementById('task');
const toneSelect = document.getElementById('tone');
const customToneInput = document.getElementById('custom-tone-input');
const customToneWarning = document.getElementById('custom-tone-warning');
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

// Detect and display hardware configuration
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
  
  // Memory (if available)
  if (navigator.deviceMemory) {
    hwMemory.textContent = `${navigator.deviceMemory} GB`;
  } else {
    hwMemory.textContent = 'N/A';
  }
  
  // GPU support check
  const gpuSupport = await evaluateGpuDelegateSupport();
  if (gpuSupport.ok) {
    hwGpu.textContent = 'WebGPU Available';
    hwGpu.classList.add('text-emerald-400');
  } else {
    hwGpu.textContent = gpuSupport.reason || 'Not Available';
    hwGpu.classList.add('text-amber-400');
  }
}

// Probe WebGPU capabilities once so we can decide whether to attempt GPU delegates.
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

      const requiredFeatures = ['shader-f16'];
      for (const feature of requiredFeatures) {
        if (!adapter.features || !adapter.features.has(feature)) {
          return { ok: false, reason: `Missing WebGPU feature: ${feature}` };
        }
      }

      return { ok: true, limits: adapter.limits || {} };
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
        case 'Custom':
          const customStyle = customToneInput.value.trim() || 'unique style';
          toneGuidelines = `\nTONE: <${customStyle}>`;
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
  if (task === 'Rewrite' && tone === 'Custom') {
    return `${instruction}\nINPUT: ${userText}\nOUTPUT:`;
  }
  
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
  const toneContainer = toneSelect.parentElement;
  if (taskSelect.value === 'Rewrite') {
    toneContainer.style.display = 'block';
  } else {
    toneContainer.style.display = 'none';
  }
});

toneSelect.addEventListener('change', () => {
  if (toneSelect.value === 'Custom') {
    customToneInput.classList.remove('hidden');
    customToneWarning.classList.remove('hidden');
    customToneInput.focus();
  } else {
    customToneInput.classList.add('hidden');
    customToneWarning.classList.add('hidden');
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
  submit.disabled = true;
  submitText.textContent = "Generating...";

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
      output.value += partialResult;
      fullResult += partialResult;
      output.scrollTop = output.scrollHeight;

      if (complete) {
        const latency = Math.round(performance.now() - start);

        metricEngine.textContent = "Base (Nano)";
        metricTime.textContent = `${latency}ms`;
        metricQuality.textContent = "Reliable";
        metricEnergy.textContent = estimateEnergy(latency);
        setDeviceStatus("Done – Base inference");

        const metrics = calculateQualityMetrics(userText, fullResult);
        displayQualityMetrics(metrics);
        
        submit.disabled = false;
        submitText.textContent = "Generate";
      }
    });

  } catch (e) {
    console.error("Generation error:", e);
    setDeviceStatus("Generation error - check console");
    statusIndicator.className = "status-dot error";
    submit.disabled = false;
    submitText.textContent = "Generate";
  }
};


if (taskSelect.value !== 'Rewrite') {
  toneContainer.style.display = 'none';
}
updateCounters();
submit.disabled = false;

// Detect hardware on page load
detectHardware();