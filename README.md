# EdgeWriter

An offline, on-device NLP toolkit for summarization, rewriting, and proofreading powered by quantized Transformer models.

## Installation & Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Geek-yash666/EdgeWriter
   cd EdgeWriter
   ```
2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install torch transformers datasets pandas numpy matplotlib nltk seaborn tqdm
   ```

## Running the Notebook

Launch Jupyter (or VS Code) from the repository root:

```bash
jupyter notebook
```

Open `notebooks/setup.ipynb` and execute the cells.

## Dataset Information

| Task                 | Dataset                | Split | Purpose                                                        |
| -------------------- | ---------------------- | ----- | -------------------------------------------------------------- |
| Summarization        | CNN/DailyMail (v3.0.0) | Train | Analyze article and summary lengths along with highlights.     |
| Grammar Checking     | GLUE - CoLA            | Train | Examine grammatical acceptability labels and sentence lengths. |
| Paraphrase Detection | GLUE - MRPC            | Train | Study sentence pair lengths and paraphrase label balance.      |

Datasets are fetched on demand via the Hugging Face `datasets` library. Ensure you have an active internet connection the first time you run the notebook so the data can download and cache locally.

## Author & Contact

- Author: Roop Yaswanth Nagabhairava
- Contact: nagabhairava.r@ufl.edu
