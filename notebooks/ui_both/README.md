# EdgeWriter - Dual Engine UI

This UI combines both models into a single interface with one server.

## Models Available

### 1. Base Model (MediaPipe)
- **Runs in browser** - No additional setup needed
- Uses WebGPU/CPU for inference
- Lightweight and fast
- Best for quick edits

### 2. Phi-3 Mini (Fine-tuned)
- **Runs on server** - GPU accelerated via llama.cpp
- Higher quality output
- Best for complex tasks

## How to Run

Simply run:
```
start_dual_ui.bat
```

Or manually:
```
python server.py
```

The server will:
1. Load the Phi-3 model
2. Serve the UI at http://127.0.0.1:8000
3. Open your browser automatically

## Requirements

### Python Dependencies
```
pip install fastapi uvicorn llama-cpp-python
```

### Model Files
- `../base_model/weights.bin` - MediaPipe model (loaded in browser)
- `../phi_model/phi3-writing-Q8.gguf` - Phi-3 model (loaded by server)

## Features

- Task selection: Rewrite, Summarize, Proofread, Paraphrase
- Tone control: Neutral, Professional, Friendly, Concise, Academic, Custom
- Real-time quality metrics
- Inference telemetry (latency, tokens, energy estimation)
- Hardware detection
- 100% local - no data sent to external servers
