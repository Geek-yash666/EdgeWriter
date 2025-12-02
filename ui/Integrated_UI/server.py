"""
EdgeWriter - Dual Engine Server
Serves both the UI and the Phi-3 model inference
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import time
import os
import webbrowser
import threading

app = FastAPI(title="EdgeWriter – Dual Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIG ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHI_MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "phi_model")
MODEL_PATH = os.path.join(PHI_MODEL_DIR, "phi3-writing-Q8.gguf")

print("=" * 50)
print("  EdgeWriter - Dual Engine Server")
print("=" * 50)
print(f"\nLoading Phi-3 Mini from: {MODEL_PATH}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_batch=512,
    n_gpu_layers=-1,
    verbose=False,
)
print("✓ Phi-3 Model loaded successfully!\n")

class Request(BaseModel):
    task: str
    tone: str = "Neutral"
    custom_tone: str = ""
    text: str

# TASK TEMPLATES

SUMMARIZE_TEMPLATE = """<|user|>
TASK: Summarize the text in 2-4 sentences, capturing the main progression of ideas.
RULES:
- Cover the beginning, middle, and end of the argument
- Combine related points for conciseness
- Do NOT add information not in the original
- Maintain factual accuracy

EXAMPLE INPUT: Advances in battery chemistry over the past decade have shifted from incremental improvements to structural innovations. Researchers now prioritize energy-dense solid-state architectures, aiming to reduce flammability while extending cycle life far beyond current lithium-ion norms. Supply-chain constraints still impede large-scale deployment, particularly in the sourcing of high-purity lithium and rare-earth stabilizers.
EXAMPLE OUTPUT: Battery development has moved from small refinements to structural innovations, with solid-state architectures prioritized for higher energy density, lower flammability, and longer life. Deployment remains limited by supply-chain constraints.

Now summarize:
{text}<|end|>
<|assistant|>"""

PROOFREAD_TEMPLATE = """<|user|>
TASK: Fix grammar, spelling, and punctuation errors.
RULES:
- Only fix errors, do NOT rewrite or paraphrase
- Keep the original wording and style
- Do NOT change facts or meaning
- Preserve the sentence structure

EXAMPLE INPUT: The system faild to start becuase of a memmory allocation error.
EXAMPLE OUTPUT: The system failed to start because of a memory allocation error.

EXAMPLE INPUT: Calibration complted; sensors returnd stable readings
EXAMPLE OUTPUT: Calibration completed; sensors returned stable readings.

Now proofread (fix only errors, keep original wording):
{text}<|end|>
<|assistant|>"""

PARAPHRASE_TEMPLATE = """<|user|>
TASK: Paraphrase while keeping similar length and one-to-one orderly meaning.
RULES:
- Use different words but keep ALL facts
- Do NOT add or remove information
- Maintain the same level of detail and structure of sentence
- Keep the same approximate length

EXAMPLE INPUT: The system failed to start due to a memory allocation error.
EXAMPLE OUTPUT: A memory allocation issue prevented the system from starting.

EXAMPLE INPUT: Calibration completed; sensors returned stable readings.
EXAMPLE OUTPUT: The calibration process finished, and the sensors showed consistent results.

Now paraphrase (use different words, keep same meaning and length):
{text}<|end|>
<|assistant|>"""

REWRITE_TEMPLATES = {
    "Neutral": """<|user|>
TASK: Rewrite the text for better clarity and readability.
TONE: Maintain neutral, clear language without strong stylistic choices.
RULES:
- Keep EVERY piece of information from the original
- Do NOT add interpretations, explanations, or new facts
- Do NOT remove ANY details
- Do NOT change the meaning

EXAMPLE INPUT: The system failed to start due to a memory allocation error.
EXAMPLE OUTPUT: The system failed to start because of a memory allocation error.

Now rewrite:
{text}<|end|>
<|assistant|>""",
    "Professional": """<|user|>
TASK: Rewrite the text in a professional tone.
TONE: Use formal, business-appropriate vocabulary. Use complete sentences and precise terminology.
RULES:
- Keep EVERY piece of information from the original
- Do NOT add or remove details
- Do NOT change the meaning

EXAMPLE INPUT: The system failed to start due to a memory allocation error.
EXAMPLE OUTPUT: The system encountered a startup failure attributable to a memory allocation error.

Now rewrite professionally:
{text}<|end|>
<|assistant|>""",
    "Friendly": """<|user|>
TASK: Rewrite the text in a friendly tone.
TONE: Use conversational, warm language. Use contractions and relatable phrasing.
RULES:
- Keep EVERY piece of information from the original
- Do NOT add or remove details
- Do NOT change the meaning

EXAMPLE INPUT: The system failed to start due to a memory allocation error.
EXAMPLE OUTPUT: The system couldn't start up because of a memory allocation error.

Now rewrite in a friendly way:
{text}<|end|>
<|assistant|>""",
    "Concise": """<|user|>
TASK: Rewrite the text to be extremely concise.
TONE: Be extremely brief. Remove unnecessary words while keeping all facts.
RULES:
- Keep ALL information from the original
- Remove filler words and redundancy
- Do NOT change the meaning

EXAMPLE INPUT: The system failed to start due to a memory allocation error.
EXAMPLE OUTPUT: System failed: memory allocation error.

Now rewrite concisely:
{text}<|end|>
<|assistant|>""",
    "Academic": """<|user|>
TASK: Rewrite the text in an academic tone.
TONE: Use scholarly vocabulary. Use formal academic sentence structures and precise terminology.
RULES:
- Keep EVERY piece of information from the original
- Do NOT add or remove details
- Do NOT change the meaning

EXAMPLE INPUT: The system failed to start due to a memory allocation error.
EXAMPLE OUTPUT: The system initialization was unsuccessful due to a memory allocation error.

Now rewrite academically:
{text}<|end|>
<|assistant|>"""
}

# === ROUTES ===

@app.get("/")
def index():
    return FileResponse(os.path.join(SCRIPT_DIR, "index.html"))

@app.get("/index.js")
def js():
    return FileResponse(os.path.join(SCRIPT_DIR, "index.js"), media_type="application/javascript")

@app.get("/health")
def health():
    return {"status": "ok", "model": "Phi-3 Mini (fine-tuned)", "engine": "dual"}

@app.post("/generate")
def generate(req: Request):
    start = time.time()
    
    task = req.task.strip()
    tone = req.tone.strip()
    text = req.text.strip()

    if task == "Summarize":
        prompt = SUMMARIZE_TEMPLATE.format(text=text)
    elif task == "Proofread":
        prompt = PROOFREAD_TEMPLATE.format(text=text)
    elif task == "Paraphrase":
        prompt = PARAPHRASE_TEMPLATE.format(text=text)
    elif task == "Rewrite":
        if tone == "Custom" and req.custom_tone:
            custom_tone = req.custom_tone.strip()
            prompt = f"""<|user|>
Rewrite the following text in a {custom_tone} style:

{text}<|end|>
<|assistant|>"""
        elif tone in REWRITE_TEMPLATES:
            prompt = REWRITE_TEMPLATES[tone].format(text=text)
        else:
            prompt = f"""<|user|>
Rewrite the following text in a {tone} style:

{text}<|end|>
<|assistant|>"""
    else:
        prompt = f"""<|user|>
Process the following text:

{text}<|end|>
<|assistant|>"""

    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.5,
        top_p=0.90,
        repeat_penalty=1.1, 
        stop=["<|end|>", "<|user|>", "<|assistant|>"],
        echo=False,
    )

    raw_result = output["choices"][0]["text"]
    result = raw_result.strip()

    for seq in ["<|end|>", "<|user|>", "<|assistant|>", "\n\n\n", "Summary:\n\n"]:
        if seq in result:
            result = result.split(seq)[0].strip()

    latency = round(time.time() - start, 2)
    
    usage = output.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    print(f"[{task}] Done in {latency}s | Tokens: {prompt_tokens}+{completion_tokens}={total_tokens} | Output: {result[:80]}{'...' if len(result)>80 else ''}")

    return {
        "text": result,
        "latency": latency,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens
        },
        "raw_output": raw_result
    }

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop\n")
    print("=" * 50)
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
