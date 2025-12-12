# EdgeWriter - Dual Engine UI

This UI combines **both** engines into a single interface served by one local server.

## 🚀 Models Available

### 1) Base Model (MediaPipe / Gemini Nano-style)
- **Runs in your browser** (WebGPU when available, CPU fallback)
- Best for fast, lightweight edits
- Loads when you click **Initialize Base Model** (and is protected from double-loading)

### 2) Phi-3 Mini (Fine-tuned, llama.cpp)
- **Runs on the local Python server** via `llama-cpp-python`
- Higher quality output for complex tasks
- **Loads lazily** (only when you first use Phi-3 for Generate/Chat)

## ▶️ How to Run

### Option 1: One-click (Windows)
Run:

```
start_dual_ui.bat
```

What it does:
- Starts the FastAPI server at `http://127.0.0.1:8000`
- Launches **Chrome first (preferred)**, otherwise **Edge**, using a **temporary profile** and GPU-friendly flags (so it won’t affect your main browser profile)
- Stops the server automatically when you close that browser window

### Option 2: Manual
From this folder:

```
python server.py
```

## 📦 Prerequisites

- **Python 3.x**
- **Browser with WebGPU**: Chrome or Edge recommended
- **Model files**:
	- `ui/nano_model_UI/weights.bin` (base model weights)
	- `ui/phi_model_UI/phi3-writing-Q8.gguf` (Phi-3 GGUF)

## 🛠️ Installation

Install Python dependencies (recommended: use the same requirements as the Phi UI):

```
pip install -r ..\phi_model_UI\requirements.txt
```

### ⚡ NVIDIA GPU acceleration (Phi-3)
For best Phi-3 performance on NVIDIA GPUs, install a CUDA-enabled `llama-cpp-python` wheel that matches your Python + CUDA.
The launcher prints an example wheel URL when it detects an NVIDIA GPU.

## ✨ Features

- **Writing tools**: Rewrite, Summarize, Proofread, Paraphrase + tone control (including Custom)
- **Chat mode**: Conversation UI for both engines
- **Telemetry**: latency/tokens estimates
- **Hardware info**: shows OS/cores/RAM and best-effort GPU name/VRAM (from `/api/gpu-info`)
- **Local-first**: all inference runs locally

## 🔧 Notes / Troubleshooting

- If you see the base model fetching `weights.bin` during initialization: that’s expected (the model must load into the browser). The UI prevents double-click loading.
- If the base model fails with a “stream ended” / “Expected 8 bytes” error, ensure `ui/nano_model_UI/weights.bin` exists and try a fresh reload.
- Phi-3 will not load at startup; it loads only after you select Phi-3 and run Generate/Chat.
