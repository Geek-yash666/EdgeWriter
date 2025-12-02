// index.js
const input = document.getElementById("input");
const output = document.getElementById("output");
const taskSelect = document.getElementById("task");
const generateBtn = document.getElementById("generate-btn");
const btnText = document.getElementById("btn-text");
const spinner = document.getElementById("spinner");
const deviceStatus = document.getElementById("device-status");
const statusIndicator = document.getElementById("status-indicator");
const loader = document.getElementById("loader");
const loaderStatus = document.getElementById("loader-status");
const progressBar = document.getElementById("progress-bar");

// Telemetry elements
const metricEngine = document.getElementById("metric-engine");
const metricTime = document.getElementById("metric-time");
const metricPromptTokens = document.getElementById("metric-prompt-tokens");
const metricCompletionTokens = document.getElementById("metric-completion-tokens");
const rawOutput = document.getElementById("raw-output");

// Counter elements
const charCount = document.getElementById("char-count");
const wordCount = document.getElementById("word-count");
const tokenEst = document.getElementById("token-est");

let isServerReady = false;

// Update counters
function updateCounts() {
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
input.addEventListener("input", updateCounts);

// Update progress bar
function updateProgress(percent, status = 'Processing...') {
  progressBar.style.width = `${percent}%`;
  loaderStatus.textContent = status;
}

// Detect hardware
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

// Check server status
async function checkServerStatus() {
  console.log("🔄 [EdgeWriter] Checking server status...");
  deviceStatus.textContent = "Checking server...";
  statusIndicator.className = "status-dot busy";
  
  try {
    console.log("📡 [EdgeWriter] Attempting to connect to http://127.0.0.1:8000/health");
    const res = await fetch("http://127.0.0.1:8000/health", {
      method: "GET",
      signal: AbortSignal.timeout(3000)
    });
    
    if (res.ok) {
      const data = await res.json();
      console.log("✅ [EdgeWriter] Server connected!", data);
      isServerReady = true;
      deviceStatus.textContent = "Ready – Phi-3 Mini Loaded";
      statusIndicator.className = "status-dot active";
      btnText.textContent = "Generate";
      generateBtn.disabled = false;
      return true;
    }
  } catch (e) {
    console.log("⚠️ [EdgeWriter] Health endpoint failed, trying generate endpoint...", e.message);
    try {
      const testRes = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: "Proofread", text: "test" }),
        signal: AbortSignal.timeout(30000)
      });
      
      if (testRes.ok) {
        console.log("✅ [EdgeWriter] Server connected via generate endpoint!");
        isServerReady = true;
        deviceStatus.textContent = "Ready – Phi-3 Mini Loaded";
        statusIndicator.className = "status-dot active";
        btnText.textContent = "Generate";
        generateBtn.disabled = false;
        return true;
      }
    } catch (e2) {
      console.error("❌ [EdgeWriter] Server not running:", e2.message);
    }
  }
  
  console.log("❌ [EdgeWriter] Server offline");
  isServerReady = false;
  deviceStatus.textContent = "Server Offline";
  statusIndicator.className = "status-dot error";
  btnText.textContent = "Server Offline - Click to Retry";
  generateBtn.disabled = false;
  return false;
}

// Generate text
async function generate() {
  if (!isServerReady) {
    console.log("🔄 [EdgeWriter] Server not ready, attempting connection...");
    loader.classList.remove('hidden');
    updateProgress(20, "Connecting to server...");
    const connected = await checkServerStatus();
    
    if (!connected) {
      updateProgress(0, "Server offline");
      loader.classList.add('hidden');
      alert("Could not connect to server.\n\nPlease ensure server.py is running:\n  cd notebooks/phi_model\n  python server.py");
      return;
    }
    loader.classList.add('hidden');
    return;
  }
  
  const text = input.value.trim();
  if (!text) {
    alert("Please enter some text to process");
    return;
  }

  const task = taskSelect.value;

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("🚀 [EdgeWriter] Starting generation...");
  console.log(`📋 Task: ${task}`);
  console.log(`📝 Input length: ${text.length} chars, ~${Math.ceil(text.split(/\s+/).length * 1.35)} tokens`);
  console.log(`📄 Input preview: "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"`);

  generateBtn.disabled = true;
  btnText.textContent = "Generating...";
  spinner.classList.remove("hidden");
  loader.classList.remove("hidden");
  updateProgress(30, "Sending to Phi-3...");
  
  deviceStatus.textContent = `Processing (${task})...`;
  statusIndicator.className = "status-dot busy";
  output.value = "";
  
  // Reset metrics
  metricEngine.textContent = "--";
  metricTime.textContent = "--";
  metricPromptTokens.textContent = "--";
  metricCompletionTokens.textContent = "--";
  rawOutput.value = "";

  const startTime = performance.now();
  console.log(`⏱️ [EdgeWriter] Request started at: ${new Date().toLocaleTimeString()}`);

  try {
    updateProgress(50, "Generating response...");
    console.log("📡 [EdgeWriter] Sending request to server...");
    
    const res = await fetch("http://127.0.0.1:8000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, text })
    });
    
    const endTime = performance.now();
    const clientLatency = ((endTime - startTime) / 1000).toFixed(2);
    
    console.log(`📥 [EdgeWriter] Response received! (Client-side latency: ${clientLatency}s)`);
    updateProgress(90, "Processing complete!");
    
    const data = await res.json();
    const latencyMs = Math.round(performance.now() - startTime);
    
    console.log(`✅ [EdgeWriter] Generation complete!`);
    console.log(`⏱️ Server latency: ${data.latency}s`);
    console.log(`📊 Output length: ${data.text.length} chars`);
    console.log(`🔢 Tokens: prompt=${data.tokens?.prompt || 0}, completion=${data.tokens?.completion || 0}, total=${data.tokens?.total || 0}`);
    console.log(`📄 Output preview: "${data.text.substring(0, 100)}${data.text.length > 100 ? '...' : ''}"`);    
    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    output.value = data.text;
    output.scrollTop = output.scrollHeight;
    
    // Update telemetry with token info
    metricEngine.textContent = "Phi-3 Mini";
    metricTime.textContent = `${data.latency}s`;
    metricPromptTokens.textContent = data.tokens?.prompt || "--";
    metricCompletionTokens.textContent = data.tokens?.completion || "--";
    rawOutput.value = data.raw_output || "(not available)";
    
    deviceStatus.textContent = "Done – Phi-3 inference";
    statusIndicator.className = "status-dot active";
    
    updateProgress(100, "Complete!");
    setTimeout(() => { loader.classList.add('hidden'); }, 500);
    
  } catch (err) {
    console.error("❌ [EdgeWriter] Generation error:", err);
    console.error("Error details:", err.message);
    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    output.value = "Error: Could not connect to local model.\n\nPlease ensure server.py is running:\n  cd notebooks/phi_model\n  python server.py";
    deviceStatus.textContent = "Connection Error";
    statusIndicator.className = "status-dot error";
    isServerReady = false;
    btnText.textContent = "Retry Connection";
    loader.classList.add('hidden');
  } finally {
    generateBtn.disabled = false;
    if (isServerReady) {
      btnText.textContent = "Generate";
    }
    spinner.classList.add("hidden");
  }
}

// Copy output
function copyOutput() {
  if (!output.value) return;
  navigator.clipboard.writeText(output.value);
}

// Paste text
async function pasteText() {
  try {
    const text = await navigator.clipboard.readText();
    input.value = text;
    updateCounts();
  } catch (e) { 
    alert("Clipboard access denied"); 
  }
}

// Clear all
function clearAll() {
  input.value = "";
  output.value = "";
  updateCounts();
  
  // Reset metrics
  metricEngine.textContent = "--";
  metricTime.textContent = "--";
  metricPromptTokens.textContent = "--";
  metricCompletionTokens.textContent = "--";
  rawOutput.value = "";
}

// Toggle token details visibility
function toggleTokenDetails() {
  const details = document.getElementById('token-details');
  const chevron = document.getElementById('token-chevron');
  details.classList.toggle('hidden');
  chevron.classList.toggle('rotate-180');
}

// Event listeners
generateBtn.addEventListener("click", generate);
input.addEventListener("keydown", e => e.ctrlKey && e.key === "Enter" && generate());

window.pasteText = pasteText;
window.clearAll = clearAll;
window.copyOutput = copyOutput;
window.toggleTokenDetails = toggleTokenDetails;

// Initialize
console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
console.log("⚡ [EdgeWriter] Initializing Phi-3 Mini UI...");
console.log("📅 Timestamp:", new Date().toLocaleString());
updateCounts();
detectHardware().then(() => {
  console.log("🖥️ [EdgeWriter] Hardware detection complete");
});
checkServerStatus().then(ready => {
  if (ready) {
    console.log("✅ [EdgeWriter] Ready to generate!");
  } else {
    console.log("⚠️ [EdgeWriter] Server not running. Start with: python server.py");
  }
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
});