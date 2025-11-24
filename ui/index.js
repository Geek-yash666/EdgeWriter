import { FilesetResolver as GenAiFilesetResolver, LlmInference } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai';

const input = document.getElementById('input');
const output = document.getElementById('output');
const submit = document.getElementById('submit');
const submitText = document.getElementById('submit-text');
const taskSelect = document.getElementById('task');
const toneSelect = document.getElementById('tone');
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

const modeBaseBtn = document.getElementById('mode-base');
const modeHybridBtn = document.getElementById('mode-hybrid');
const modeWarning = document.getElementById('mode-warning');

const MEDIAPIPE_MODEL = 'weights.bin';

let mediapipeLLM = null;
let gpuLLM = null;
let selectedMode = null;
let isInitialized = false;

// === Character/Word Counter ===
function updateCounters() {
  const text = input.value;
  const chars = text.length;
  const words = text.trim() ? text.trim().split(/\s+/).filter(w => w.length > 0).length : 0;
  
  charCount.textContent = `${chars} characters`;
  wordCount.textContent = `${words} words`;
}

input.addEventListener('input', updateCounters);

// === Progress Bar Updater ===
function updateProgress(percent, status = 'Processing...') {
  progressBar.style.width = `${percent}%`;
  loaderStatus.textContent = status;
}

// === Quality Metrics Calculator ===
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

// === Task-specific generation parameters (for GPU only) ===
const TASK_PARAMS = {
  'Rewrite': {
    temperature: 0.7,
    topK: 50,
    maxTokens: 512,
  },
  'Summarize': {
    temperature: 0.3,
    topK: 40,
    maxTokens: 256,
  },
  'Proofread': {
    temperature: 0.2,
    topK: 30,
    maxTokens: 512,
  },
  'Paraphrase': {
    temperature: 0.6,
    topK: 45,
    maxTokens: 512,
  }
};

// === Get parameters for current task ===
function getTaskParams(task) {
  return TASK_PARAMS[task] || TASK_PARAMS['Rewrite'];
}

async function detectHardware() {
  const platform = navigator.platform || 'Unknown';
  const cores = navigator.hardwareConcurrency || 'Unknown';
  
  let memory = 'Unknown';
  if (navigator.deviceMemory) {
    memory = `${navigator.deviceMemory} GB`;
  }
  
  let gpu = 'Not Available';
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        gpu = 'WebGPU Available';
      }
    } catch (e) {
      gpu = 'WebGPU Not Supported';
    }
  }
  
  document.getElementById('hw-platform').textContent = platform;
  document.getElementById('hw-cores').textContent = cores;
  document.getElementById('hw-memory').textContent = memory;
  document.getElementById('hw-gpu').textContent = gpu;
}

// === Energy Estimation ===
function estimateEnergy(latencyMs, mode) {
  const cpuPowerW = 0.5;
  const gpuPowerW = 3.0;
  
  const timeSeconds = latencyMs / 1000;
  
  let energyWs;
  if (mode === 'hybrid') {
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
      return `TASK: Summarize the text in maximum 3 sentences.\nRULES:\n- Keep all key facts and details\n- Do NOT add information not in the original\n- Do NOT remove important details\n- Maintain factual accuracy\nOutput only the summary:`;
    case 'Proofread': 
      return `TASK: Fix grammar, spelling, and punctuation errors.\nRULES:\n- Only fix errors, do NOT rewrite\n- Keep the original wording and style\n- Do NOT change facts or meaning\n- Preserve the sentence structure\nOutput only the corrected text:`;
    case 'Paraphrase': 
      return `TASK: Paraphrase while keeping similar length and meaning.\nRULES:\n- Use different words but keep ALL facts\n- Do NOT add or remove information\n- Maintain the same level of detail\n- Keep the same approximate length\nOutput only the paraphrased text:`;
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
  if (task === 'Rewrite' && fewShotExamples['Rewrite'][tone]) {
    examples = fewShotExamples['Rewrite'][tone];
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

// === MODE SELECTION ===
window.selectMode = (mode) => {
  selectedMode = mode;
  
  modeBaseBtn.classList.remove('border-blue-500');
  modeHybridBtn.classList.remove('border-purple-500');
  
  if (mode === 'base') {
    modeBaseBtn.classList.add('border-blue-500');
    modeWarning.classList.add('hidden');
    submitText.textContent = 'Initialize Base Model';
  } else {
    modeHybridBtn.classList.add('border-purple-500');
    modeWarning.classList.remove('hidden');
    submitText.textContent = 'Initialize Hybrid Model';
  }
  
  submit.disabled = false;
  isInitialized = false;
  
  deviceStatus.textContent = "Mode selected – ready to initialize";
  statusIndicator.className = "status-dot";
};

function isBetter(a, b) {
  if (!a || a.length < 5) return false;
  if (a.includes("User:") || a.includes("Assistant:")) return false;
  if (b.includes("User:") || b.includes("Assistant:")) return true;
  return a.length > b.length * 0.8;
}

async function init() {
  if (!selectedMode) {
    alert("Please select an inference mode first (Base or Hybrid)");
    return false;
  }

  submit.disabled = true;
  submitText.textContent = "Loading Models...";
  deviceStatus.textContent = "Initializing...";
  statusIndicator.className = "status-dot busy";

  try {
    updateProgress(20, "Loading MediaPipe base model...");
    deviceStatus.textContent = "Loading MediaPipe base model...";
    const genaiFileset = await GenAiFilesetResolver.forGenAiTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm"
    );
    
    updateProgress(50, "Initializing model...");
    mediapipeLLM = await LlmInference.createFromOptions(genaiFileset, {
      baseOptions: { modelAssetPath: MEDIAPIPE_MODEL, delegate: 'CPU' },
      maxTokens: 512,
      temperature: 0.5,
      topK: 40,
    });

    updateProgress(70, "Base model ready");

    if (selectedMode === 'hybrid') {
      updateProgress(75, "Loading fine-tuned model weights...");
      deviceStatus.textContent = "Loading custom Phi-3 weights...";
      try {
        const { CreateMLCEngine } = await import("https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.79/lib/index.min.js");

        // Custom model configuration
        const customModelRecord = {
          model: "http://localhost:8000/phi3-mlc-output",
          model_id: "phi3-writing-custom",
          model_lib: "D:\\OneDrive - University of Florida\\Courses\\Fall 2025\\Codes\\ML\\Project\\EdgeWriter\\ui\\phi3-mlc-output\\phi3-webgpu.wasm",
          //model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_48/Phi-3-mini-4k-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
          vram_required_MB: 2048,
          low_resource_required: false,
        };

        gpuLLM = await CreateMLCEngine(
          "phi3-writing-custom",
          {
            appConfig: {
              model_list: [customModelRecord],
              useIndexedDBCache: false
            },
            initProgressCallback: (report) => {
              console.log("Load progress:", report);
              const progress = 75 + (report.progress * 25);
              updateProgress(progress, `Loading: ${Math.round(report.progress * 100)}%`);
              deviceStatus.textContent = `Loading weights: ${Math.round(report.progress * 100)}%`;
            },
          }
        );

        updateProgress(100, "Ready!");
        deviceStatus.textContent = "Ready – Hybrid Mode (Fine-tuned Phi-3)";
        statusIndicator.className = "status-dot active";
      } catch (e) {
        console.error("GPU model loading failed:", e);
        console.error("Full error stack:", e.stack);
        alert(`Failed to load custom model: ${e.message}\nFalling back to Base Mode.`);
        selectedMode = 'base';
        modeBaseBtn.classList.add('border-blue-500');
        modeHybridBtn.classList.remove('border-purple-500');
        updateProgress(100, "Ready (Base Mode)");
        deviceStatus.textContent = "Ready – Base Mode (CPU Only)";
        statusIndicator.className = "status-dot active";
      }
    } else {
      updateProgress(100, "Ready!");
      deviceStatus.textContent = "Ready – Base Mode (CPU Only)";
      statusIndicator.className = "status-dot active";
    }

    isInitialized = true;
    submit.disabled = false;
    submitText.textContent = "Generate";
    return true;

  } catch (e) {
    console.error("Initialization failed:", e);
    updateProgress(0, "Initialization failed");
    deviceStatus.textContent = "Initialization failed";
    statusIndicator.className = "status-dot error";
    alert("Failed to initialize models. Check console for details.");
    submit.disabled = false;
    submitText.textContent = "Retry Initialization";
    return false;
  }
}

submit.onclick = async () => {
  if (!isInitialized) {
    const success = await init();
    if (!success || !mediapipeLLM) return;
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
  
  const taskParams = getTaskParams(task);

  output.value = "";
  resetQualityMetrics();
  loader.classList.remove('hidden');
  submit.disabled = true;

  metricEngine.textContent = "--";
  metricTime.textContent = "--";
  metricQuality.textContent = "--";
  metricEnergy.textContent = "--";

  const start = performance.now();
  let cpuResult = "";

  try {
    const taskLabel = task === 'Rewrite' ? `${task} (${tone})` : task;
    
    updateProgress(30, `Generating with base model...`);
    deviceStatus.textContent = `Generating (${taskLabel})...`;
    
    cpuResult = await mediapipeLLM.generateResponse(fullPrompt);
    output.value = cpuResult.trim();

    updateProgress(60, "Base generation complete");

    const latency = Math.round(performance.now() - start);
    
    metricEngine.textContent = "Draft Model";
    metricTime.textContent = `${latency}ms`;
    metricQuality.textContent = "Reliable";
    metricEnergy.textContent = estimateEnergy(latency, 'base');

    if (selectedMode === 'hybrid' && gpuLLM) {
      updateProgress(70, `Enhancing with GPU...`);
      deviceStatus.textContent = `Enhancing with GPU (${taskLabel})...`;

      const res = await gpuLLM.chat.completions.create({
        messages: [{ role: "user", content: fullPrompt }],
        temperature: taskParams.temperature,
        max_gen_len: taskParams.maxTokens,
        top_k: taskParams.topK,
      });

      const gpuResult = res.choices[0].message.content.trim();
      const totalLatency = Math.round(performance.now() - start);

      updateProgress(90, "Finalizing...");

      if (isBetter(gpuResult, cpuResult)) {
        output.value = gpuResult;
        metricEngine.textContent = "GPU (Q8 Custom)";
        metricTime.textContent = `${totalLatency}ms`;
        metricQuality.textContent = "Enhanced";
        metricQuality.className = "text-sm font-semibold text-emerald-400";
        metricEnergy.textContent = estimateEnergy(totalLatency, 'hybrid');
        deviceStatus.innerHTML = '<span style="color:#4ade80">Enhanced with GPU</span>';
        
        const metrics = calculateQualityMetrics(userText, gpuResult);
        displayQualityMetrics(metrics);
      } else {
        deviceStatus.textContent = "Base model output preferred";
        
        const metrics = calculateQualityMetrics(userText, cpuResult);
        displayQualityMetrics(metrics);
      }
    } else {
      deviceStatus.textContent = "Done – Base model only";
      
      const metrics = calculateQualityMetrics(userText, cpuResult);
      displayQualityMetrics(metrics);
    }

    updateProgress(100, "Complete!");
  } catch (e) {
    console.error("Generation error:", e);
    updateProgress(0, "Error occurred");
    deviceStatus.textContent = "Generation error";
    statusIndicator.className = "status-dot error";
    if (cpuResult) output.value = cpuResult;
  } finally {
    setTimeout(() => {
      loader.classList.add('hidden');
      updateProgress(0, "");
    }, 500);
    submit.disabled = false;
  }
};

// Initialize
detectHardware();

const toneContainer = document.getElementById('tone-container');
if (taskSelect.value !== 'Rewrite') {
  toneContainer.style.display = 'none';
}
updateCounters();
submit.disabled = true;