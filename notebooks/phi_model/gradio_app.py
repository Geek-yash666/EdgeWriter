import gradio as gr
from pathlib import Path
import time

try:
    from llama_cpp import Llama
except ImportError as exc: 
    raise ImportError(
        "llama-cpp-python is required. Install it with `pip install llama-cpp-python` before running this demo."
    ) from exc

MODEL_PATH = Path("phi3-writing-Q8.gguf")
_llm = None

# === TASK TEMPLATES ===

SUMMARIZE_TEMPLATE = """<|user|>
TASK: Summarize the text in 2-4 sentences, capturing the main progression of ideas.
RULES:
- Cover the beginning, middle, and end of the argument
- Combine related points for conciseness
- Do NOT add information not in the original
- Maintain factual accuracy
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

REWRITE_TEMPLATE = """<|user|>
TASK: Rewrite the text for better clarity and readability.
RULES:
- Keep EVERY piece of information from the original
- Do NOT add interpretations, explanations, or new facts
- Do NOT remove ANY details
- Do NOT change the meaning
- Improve sentence structure and flow

EXAMPLE INPUT: The system failed to start due to a memory allocation error.
EXAMPLE OUTPUT: The system failed to start because of a memory allocation error.

Now rewrite for clarity:
{text}<|end|>
<|assistant|>"""

def load_llm():
    global _llm
    if _llm is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Could not find model file at {MODEL_PATH.resolve()}")
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            verbose=False,
        )
    return _llm

def process_text(text, task):
    """Generate a response from the local Phi-3 model."""
    if not text or not text.strip():
        return "Please enter some text to process.", "--", ""
    
    start = time.time()
    llm = load_llm()
    text = text.strip()
    
    # Build prompt based on task
    if task == "Summarize":
        prompt = SUMMARIZE_TEMPLATE.format(text=text)
    elif task == "Proofread":
        prompt = PROOFREAD_TEMPLATE.format(text=text)
    elif task == "Paraphrase":
        prompt = PARAPHRASE_TEMPLATE.format(text=text)
    elif task == "Rewrite":
        prompt = REWRITE_TEMPLATE.format(text=text)
    else:
        prompt = f"""<|user|>
Process the following text:

{text}<|end|>
<|assistant|>"""
    
    output = llm(
        prompt=prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.90,
        stop=["<|end|>", "<|user|>", "<|assistant|>"],
    )
    
    result = output["choices"][0]["text"].strip()
    
    # Cleanup
    for seq in ["<|end|>", "<|user|>", "<|assistant|>", "\n\n\n", "Summary:\n\n"]:
        if seq in result:
            result = result.split(seq)[0].strip()
    
    latency = round(time.time() - start, 2)
    
    # Extract token usage info
    usage = output.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    raw_output = output["choices"][0]["text"]
    
    token_info = f"""📊 Token Usage:
• Prompt tokens: {prompt_tokens}
• Completion tokens: {completion_tokens}
• Total tokens: {total_tokens}

🔤 Raw Generated Text (before cleanup):
{raw_output}"""
    
    return result, f"{latency}s", token_info

with gr.Blocks(title="EdgeWriter - Phi-3 Writing Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📝 EdgeWriter - Phi-3 Writing Assistant")
    
    with gr.Row():
        task_select = gr.Dropdown(
            choices=["Summarize", "Proofread", "Paraphrase", "Rewrite"],
            value="Proofread",
            label="Task"
        )
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter your text here...",
                lines=8,
            )
            submit_button = gr.Button("✨ Generate", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="Output", interactive=False, lines=8)
            latency_display = gr.Textbox(label="Processing Time", interactive=False)
    
    with gr.Accordion("🔍 Token Details", open=False):
        token_info_display = gr.Textbox(label="Token Generation Info", interactive=False, lines=10)

    submit_button.click(
        process_text,
        inputs=[user_input, task_select],
        outputs=[output_text, latency_display, token_info_display]
    )
    
    user_input.submit(
        process_text,
        inputs=[user_input, task_select],
        outputs=[output_text, latency_display, token_info_display]
    )
demo.launch()