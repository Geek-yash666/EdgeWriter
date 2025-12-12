"""
EdgeWriter - Dual Engine Server
Serves both the UI and the Phi-3 model inference
"""
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama
from typing import List, Optional
import uvicorn
import time
import os
import webbrowser
import threading
import subprocess
import sys
import psutil
import re
import tempfile
import shutil
import atexit

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
PHI_MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "phi_model_UI")
MODEL_PATH = os.path.join(PHI_MODEL_DIR, "phi3-writing-Q8.gguf")
NANO_UI_DIR = os.path.join(SCRIPT_DIR, "..", "nano_model_UI")

# === Browser launch===
URL = "http://127.0.0.1:8000"
temp_profile = tempfile.mkdtemp(prefix="edgewriter_gpu_force_")
browser_process = None
_cleanup_done = False


def cleanup_browser_profile():
    global browser_process, _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    try:
        if browser_process and browser_process.poll() is None:
            try:
                browser_process.terminate()
                browser_process.wait(timeout=3)
            except Exception:
                try:
                    browser_process.kill()
                except Exception:
                    pass
        shutil.rmtree(temp_profile, ignore_errors=True)
    except Exception as e:
        print(f"Warning: browser cleanup failed: {e}")


atexit.register(cleanup_browser_profile)


def find_browser_executable():
    """Prefer Chrome, then Edge, with common Windows install paths."""
    if sys.platform == "win32":
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path

        # PATH fallback (prefer chrome.exe)
        for exe in ["chrome.exe", "msedge.exe"]:
            try:
                subprocess.check_output(["where", exe], stderr=subprocess.DEVNULL)
                return exe
            except Exception:
                pass
        return None

    # non-windows fallback
    for exe in ["google-chrome", "chrome", "chromium", "microsoft-edge"]:
        try:
            subprocess.check_output(["which", exe], stderr=subprocess.DEVNULL)
            return exe
        except Exception:
            pass
    return None


def launch_browser_with_gpu_force():
    """Launch Chrome/Edge with a temporary profile and GPU flags."""
    global browser_process
    browser_path = find_browser_executable()
    if not browser_path:
        return False

    browser_flags = [
        "--new-window",
        f"--user-data-dir={temp_profile}",
        "--force-high-performance-gpu",
        "--ignore-gpu-blocklist",
        "--enable-features=Vulkan",
        "--disable-features=UseChromeOSDirectVideoDecoder",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-software-rasterizer",
        "--disable-background-mode",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-extensions",
        "--no-service-autorun",
        URL,
    ]

    cmd = [browser_path, *browser_flags]
    try:
        browser_process = subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        print(f"Error launching browser: {e}")
        return False


def monitor_browser_and_stop(server: "uvicorn.Server"):
    """Stop uvicorn when the launched browser window closes."""
    global browser_process
    if not browser_process:
        return
    try:
        print("Monitoring browser process (server will stop when browser closes)...")
        browser_process.wait()
    except Exception as e:
        print(f"Browser monitoring error: {e}")
    finally:
        print("Browser closed - shutting down server...")
        try:
            server.should_exit = True
        except Exception:
            pass
        cleanup_browser_profile()


# === Phi-3 lazy-load state ===
_llm: Optional[Llama] = None
_llm_lock = threading.Lock()


def get_llm() -> Llama:
    """Load and return the Phi-3 Llama instance on first use."""
    global _llm
    if _llm is not None:
        return _llm

    with _llm_lock:
        if _llm is not None:
            return _llm

        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(f"Phi-3 model file not found: {MODEL_PATH}")

        print(f"Loading Phi-3 Mini on-demand from: {MODEL_PATH}")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False,
        )
        print("✓ Phi-3 Model loaded successfully!\n")
        return _llm


def get_gpu_info():
    """Detect available GPUs on the system (best-effort)."""
    gpus = []
    try:
        # nvidia-smi for NVIDIA GPUs
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            for line in output.strip().split("\n"):
                parts = line.split(",")
                if len(parts) >= 1:
                    gpus.append(
                        {
                            "name": parts[0].strip(),
                            "type": "NVIDIA",
                            "memory": parts[1].strip() if len(parts) > 1 else "Unknown",
                        }
                    )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Fallback to WMIC on Windows for all GPUs
        if sys.platform == "win32":
            try:
                output = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "Name"],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                )
                lines = output.strip().split("\n")
                for line in lines:
                    name = line.strip()
                    if name and "Name" not in name and not any(g["name"] == name for g in gpus):
                        gpu_type = (
                            "Integrated"
                            if any(x in name.upper() for x in ["INTEL", "AMD RADEON(TM) GRAPHICS"])
                            else "Dedicated"
                        )
                        gpus.append({"name": name, "type": gpu_type, "memory": "Unknown"})
            except Exception:
                pass
    except Exception as e:
        print(f"Error detecting GPUs: {e}")

    return gpus


def get_system_info():
    """Return basic system info including RAM in GB."""
    ram_gb = None
    try:
        mem = psutil.virtual_memory()
        if mem and mem.total:
            ram_gb = round(mem.total / (1024**3))
    except Exception:
        pass

    # Fallback to WMIC if psutil not sufficient
    if ram_gb is None and sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["wmic", "OS", "get", "TotalVisibleMemorySize"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            lines = [
                ln.strip()
                for ln in output.splitlines()
                if ln.strip() and "TotalVisibleMemorySize" not in ln
            ]
            if lines:
                kb = float(lines[0])
                ram_gb = round((kb * 1024) / (1024**3))
        except Exception:
            pass

    return {"ramGB": ram_gb}


def _weights_file_path() -> str:
    return os.path.join(NANO_UI_DIR, "weights.bin")

class Request(BaseModel):
    task: str
    tone: str = "Neutral"
    custom_tone: str = ""
    text: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]

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
    return {
        "status": "ok",
        "model": "Phi-3 Mini (fine-tuned)",
        "engine": "dual",
        "phiLoaded": _llm is not None,
    }


@app.get("/api/gpu-info")
def gpu_info():
    gpus = get_gpu_info()
    system = get_system_info()
    return {"gpus": gpus, **system}


@app.get("/nano_model_UI/weights.bin")
def nano_weights(range_header: Optional[str] = Header(None, alias="Range")):
    """Serve weights.bin with proper HTTP Range support (MediaPipe requires this)."""
    weights_path = _weights_file_path()
    if not os.path.isfile(weights_path):
        raise HTTPException(status_code=404, detail="weights.bin not found")

    file_size = os.path.getsize(weights_path)
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=31536000, immutable",
    }

    if not range_header:
        def iter_full():
            with open(weights_path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        headers = {**common_headers, "Content-Length": str(file_size)}
        return StreamingResponse(iter_full(), media_type="application/octet-stream", headers=headers)

    match = re.match(r"^bytes=(\d*)-(\d*)$", range_header.strip())
    if not match:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    start_str = match.group(1)
    end_str = match.group(2)

    # bytes=START-END
    # bytes=START-
    # bytes=-SUFFIX_LEN
    if start_str == "" and end_str == "":
        raise HTTPException(status_code=416, detail="Invalid Range header")

    if start_str == "" and end_str != "":
        # suffix range: last N bytes
        suffix_len = int(end_str)
        if suffix_len <= 0:
            raise HTTPException(status_code=416, detail="Invalid Range header")
        start = max(file_size - suffix_len, 0)
        end = file_size - 1
    else:
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1

    if start >= file_size or end < start:
        return StreamingResponse(
            iter(()),
            status_code=416,
            media_type="application/octet-stream",
            headers={**common_headers, "Content-Range": f"bytes */{file_size}"},
        )

    end = min(end, file_size - 1)
    length = end - start + 1

    def iter_range():
        with open(weights_path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        **common_headers,
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Content-Length": str(length),
    }
    return StreamingResponse(
        iter_range(),
        status_code=206,
        media_type="application/octet-stream",
        headers=headers,
    )

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

    llm = get_llm()
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


CHAT_SYSTEM_PROMPT = """<|system|>
You are EdgeWriter Chat. Respond concisely and follow the user's instructions directly.
Keep responses under 200 tokens unless explicitly asked for more.
Treat prior turns as context; do not repeat them.
<|end|>"""


def build_chat_prompt(messages: List[ChatMessage]):
    trimmed = messages[-12:]
    parts = [CHAT_SYSTEM_PROMPT]
    for msg in trimmed:
        role = "assistant" if (msg.role or "").lower().strip() == "assistant" else "user"
        content = (msg.content or "").strip()
        parts.append(f"<|{role}|>\n{content}\n<|end|>")
    parts.append("<|assistant|>")
    return "\n".join(parts)


@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    prompt = build_chat_prompt(req.messages)

    llm = get_llm()
    output = llm(
        prompt,
        max_tokens=2048,
        temperature=0.5,
        top_p=0.9,
        repeat_penalty=1.05,
        stop=["<|end|>", "<|user|>", "<|assistant|>"],
        echo=False,
    )

    raw_result = output["choices"][0]["text"]
    result = raw_result.strip()
    for seq in ["<|end|>", "<|user|>", "<|assistant|>"]:
        if seq in result:
            result = result.split(seq)[0].strip()

    latency = round(time.time() - start, 2)
    usage = output.get("usage", {})

    return {
        "text": result,
        "latency": latency,
        "tokens": {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0),
        },
        "raw_output": raw_result,
    }


# NOTE: This mount is intentionally placed AFTER the explicit weights.bin route
# so MediaPipe range requests use the handler above.
if os.path.isdir(NANO_UI_DIR):
    app.mount("/nano_model_UI", StaticFiles(directory=NANO_UI_DIR), name="nano_model_UI")

def open_browser():
    time.sleep(1.5)
    # Keep as a fallback; primary launch uses temp-profile Chrome/Edge
    webbrowser.open(URL)

if __name__ == "__main__":
    print("=" * 50)
    print("  EdgeWriter - Dual Engine Server")
    print("=" * 50)
    print(f"\nPhi-3 model path: {MODEL_PATH}")
    print("Phi-3 will load on first /generate or /chat request.\n")
    print("Starting server at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop\n")
    print("=" * 50)
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Try to launch Chrome/Edge with temp profile + GPU flags
    if launch_browser_with_gpu_force():
        threading.Thread(target=monitor_browser_and_stop, args=(server,), daemon=True).start()
    else:
        # Fallback: open default browser
        threading.Thread(target=open_browser, daemon=True).start()

    server.run()