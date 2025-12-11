# ✨ EdgeWriter AI

**EdgeWriter** is a private, local-first AI writing assistant powered by Microsoft's **Phi-3 Mini** model. It runs entirely on your machine, ensuring your data never leaves your computer.

Designed for speed and privacy, it offers two distinct user interfaces: a modern **Web App** (FastAPI + HTML/JS) and a rapid-prototyping  **Gradio Interface** .

## 🚀 Features

* **🔒 100% Local & Private** : No API keys, no cloud data transfer. Works offline.
* **⚡ Dual Interfaces** :
* **Custom Web UI** : A beautiful, glassmorphism-styled web app with real-time telemetry.
* **Gradio UI** : A sturdy, standard interface for quick testing and sharing.
* **✍️ Writing Tools** :
* **Summarize** : Condense long text into key points.
* **Proofread** : Fix grammar, spelling, and punctuation errors.
* **Paraphrase** : Reword text while preserving meaning.
* **Rewrite** : Change the tone (Professional, Friendly, Academic, Concise, or Custom).
* **💬 Chat Mode** : Interactive conversation with context awareness.
* **🚀 Hardware Acceleration** : Automatic NVIDIA GPU detection and offloading via `llama.cpp`.

## 📂 File Structure

```
EdgeWriter/
├── phi3-writing-Q8.gguf      # The quantized Phi-3 Mini model file (Required)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── [ Web UI Files ]
├── server.py                 # FastAPI backend server
├── index.html                # Frontend HTML (Tailwind CSS)
├── index.js                  # Frontend Logic
├── start_web_ui.bat          # One-click launcher for Web UI
│
└── [ Gradio UI Files ]
    ├── gradio_app.py         # Gradio interface script
    └── start_gradio.bat      # One-click launcher for Gradio
```

## 🛠️ Prerequisites

1. **Python 3.10+** : Ensure Python is installed and added to your system PATH.
2. **NVIDIA GPU (Recommended)** : For fast generation. It runs on CPU, but significantly slower.
3. **The Model File** : You must have the `phi3-writing-Q8.gguf` file placed in the root directory.

## 📦 Installation

1. **Clone/Download this repository.**
2. **Install Dependencies:**
   Open a terminal/command prompt in the folder and run:

   ```
   pip install -r requirements.txt
   ```

   > **⚡ GPU Acceleration Tip:**
   > To get maximum speed on NVIDIA GPUs, you must install the CUDA-enabled version of `llama-cpp-python`.
   >

   > **For Windows (cuBLAS):**
   >

   > ```
   > pip install --upgrade --force-reinstall --no-cache-dir [https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl](https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp310-cp310-win_amd64.whl)
   > ```
   >

   > *(Note: Adjust the URL based on your Python version and CUDA version).*
   >

## ▶️ How to Run

You can run either interface using the provided batch files (Windows) or via terminal.

### Option 1: The Modern Web UI (Recommended)

This launches a custom FastAPI server and a sleek frontend.

* **Windows** : Double-click `start_web_ui.bat`.
* **Manual** :

```
  python server.py
```

* **Access** : The script attempts to open your browser automatically. If not, go to `http://127.0.0.1:8000`.

### Option 2: The Gradio Interface

Best for debugging or if you prefer the standard Gradio look.

* **Windows** : Double-click `start_gradio.bat`.
* **Manual** :

```
  python gradio_app.py
```

## **SCREENSHOTS**
