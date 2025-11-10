    

# EdgeWriter

An offline, on-device AI writing assistant for summarization, rewriting, proofreading, and paraphrasing. EdgeWriter uses a dual-model architecture with speculative decoding to provide fast, accurate text generation that runs entirely on your device - no internet required.

## Project Overview

EdgeWriter implements a **speculative decoding architecture** that adapts to your hardware:

- **CPU-only mode**: Uses a lightweight draft model for fast token generation (<5 seconds, ~2GB RAM)
- **GPU-accelerated mode**: Combines the draft model with a fine-tuned base model for higher accuracy through token verification.

This architecture enables efficient on-device inference across a range of hardware configurations.

## Architecture

### Base Model

- **Foundation**: Microsoft Phi-3-mini-4k-instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: Q4 quantization for efficient deployment
- **Interface**: Gradio web UI for interactive testing

### Draft Model

- **Type**: Small, highly optimized language model
- **Implementation**: JavaScript (MediaPipe LLM Inference)
- **Deployment**: Browser-based, runs via WebAssembly
- **Performance**: <5 sec generation, ~2GB RAM usage

### Speculative Decoding Pipeline

1. **CPU-only systems**: Draft model generates tokens directly for final output
2. **GPU-enabled systems**: Draft model generates candidate tokens → Base model verifies and refines → Enhanced accuracy output

## Training Details

### Datasets

All datasets were combined into a unified training corpus:

| Task          | Dataset       | Purpose                        |
| ------------- | ------------- | ------------------------------ |
| Summarization | CNN/DailyMail | Generate concise summaries     |
| Editing       | CoEDIT        | Text editing and rewriting     |
| Grammar       | JFLEG         | Grammar correction             |
| Paraphrasing  | PAWS          | Semantic paraphrase generation |

### Model Performance

The fine-tuned base model was evaluated using standard NLP metrics:

**ROUGE Scores:**

- ROUGE-1: 0.7886
- ROUGE-2: 0.6238
- ROUGE-L: 0.7478
- ROUGE-Lsum: 0.7482

**BLEU Score:**

- BLEU: 0.5905
- Brevity Penalty: 1.0
- Length Ratio: 1.015

## Quick Start

### Running the Draft Model (Browser-based)

1. **Navigate to the draft model directory:**

   ```bash
   cd EdgeWriter/notebooks/draft_model
   ```
2. **Ensure `weights.bin` is present** in this directory.
3. **Start a local web server:**

   ```bash
   python -m http.server 8000
   ```
4. **Open in browser:**
   Navigate to `http://localhost:8000` in your web browser.

The draft model runs entirely in your browser using JavaScript and WebAssembly—no Python installation needed for inference!

### Running the Base Model (GPU-accelerated)

1. **Set up the Python environment** (see Installation section below).
2. **Open the base model notebook:**

   ```bash
   jupyter lab
   ```

   Navigate to `notebooks/base_model/Phi3_finetuned_base_model.ipynb`.
3. **Run the Gradio cell** to launch the interactive interface.
4. **Access via browser** at the provided local URL (typically `http://localhost:7860`).

### Example Output

Here's an example of the base model performing a proofreading task:

![Base Model Output](docs/draft_proofread.png)

## 💻 Installation & Environment Setup

### Prerequisites

- Python 3.11+
- CUDA 12+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB recommended for base model training)

### Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Geek-yash666/EdgeWriter
   cd EdgeWriter
   ```
2. **Option A: Using Conda (Recommended)**

   ```bash
   conda env create -f environment.yml
   conda activate edgewriter
   ```
3. **Option B: Using pip**

   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
EdgeWriter/
├── notebooks/
│   ├── base_model/              # Fine-tuned Phi-3 model
│   │   └── Phi3_finetuned_base_model.ipynb
│   └── draft_model/             # Browser-based draft model
│       ├── index.html           # Web UI
│       ├── index.js             # JavaScript inference
│       └── weights.bin          # Model weights
├── data/                        # Dataset processing notebooks
|-- docs/                        # Architecture/ Inference Screenshots
├── results/                     # Training outputs and metrics
├── environment.yml              # Conda environment
├── requirements.txt             # Pip dependencies
└── README.md
```

## Model Development Workflow

1. **Data Processing**: Combine and preprocess datasets from multiple sources
2. **Base Model Fine-tuning**: Apply LoRA to Phi-3-mini-4k-instruct
3. **Model Merging**: Merge LoRA adapters with base weights
4. **Quantization**: Apply Q4 quantization for deployment
5. **Evaluation**: Test with ROUGE and BLEU metrics
6. **Draft Model Optimization**: Create lightweight model for browser deployment
7. **Integration**: Implement speculative decoding pipeline

## 🛠️ Technologies Used

- **Training**: PyTorch, Hugging Face Transformers, PEFT (LoRA)
- **Inference**: MediaPipe LLM Inference (JavaScript), Gradio
- **Quantization**: bitsandbytes, GGUF format
- **Datasets**: Hugging Face Datasets library
- **UI**: HTML/CSS/JavaScript, Gradio

## 👤 Author & Contact

- **Author**: Roop Yaswanth Nagabhairava
- **Email**: nagabhairava.r@ufl.edu
- **Institution**: University of Florida

## 📄 License

This project is licensed under the terms specified in the LICENSE file.
