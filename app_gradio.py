import os
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------- Device & dtype (MPS-safe) --------------------
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "mps" else (torch.bfloat16 if DEVICE == "cuda" else torch.float32)

# -------------------- Configuration --------------------
BASE_ID = os.environ.get("BASE_ID", "meta-llama/Llama-3.2-1B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "phishsense_lora_adapter")
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
PORT = int(os.environ.get("PORT", "7860"))
SHARE = os.environ.get("SHARE", "false").lower() in {"1", "true", "yes"}

# -------------------- Load model (no device_map='auto' for MPS) --------------------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, token=HF_TOKEN, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        token=HF_TOKEN,
        dtype=DTYPE,   # torch_dtype deprecated; use dtype
    )

    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# -------------------- Inference --------------------
PROMPT_PREFIX = (
    "Classify the following text as phishing or not. "
    "Respond with 'TRUE' or 'FALSE' (exactly one word):\n\n"
)

def predict_single(email_text: str) -> str:
    if not email_text or not email_text.strip():
        return "Please paste an email."

    prompt = f"{PROMPT_PREFIX}{email_text}\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    # Ensure inputs are on the same device as the model (fixes MPS placeholder error)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Robust parsing to find first TRUE/FALSE after "Answer:"
    raw = text.split("Answer:")[-1].strip()
    cleaned = re.sub(r"[^A-Za-z ]", " ", raw)
    m = re.search(r"\b(TRUE|FALSE)\b", cleaned, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return f"Uncertain (raw: {raw})"

# -------------------- UI --------------------
with gr.Blocks(title="PhishSense — LoRA (Gradio, MPS-safe)") as demo:
    gr.Markdown(
        "# 📧 PhishProof — Phishing Classifier\n"
        "Paste an email and get `TRUE` (phishing) or `FALSE` (not phishing)."
    )
    with gr.Row():
        email = gr.Textbox(
            label="Email text",
            lines=12,
            placeholder="Paste the email body here..."
        )
        result = gr.Label(label="Prediction")

    gr.Examples(
        examples=[
            ["Urgent: Your account has been flagged for suspicious activity. Please log in immediately."],
            ["Hi Jason, thanks for your help last week. Here are the meeting notes attached."],
            ["We noticed unusual sign-in attempts. Verify your identity here: http://example-login-security.com"]
        ],
        inputs=[email],
    )

    btn = gr.Button("Classify")
    btn.click(fn=predict_single, inputs=[email], outputs=[result])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=PORT, share=SHARE)
