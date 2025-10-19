import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
import re

MODEL_ID = "AcuteShrewdSecurity/Llama-Phishsense-1B"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base_model, MODEL_ID)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def predict_single(email_text: str) -> str:
    if not email_text.strip():
        return "Please paste an email."
    prompt = (
        "Classify the following text as phishing or not. "
        "Respond with 'TRUE' or 'FALSE':\n\n"
        f"{email_text}\nAnswer:"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
    )
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.01,
            do_sample=False
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    raw = text.split("Answer:")[-1].strip()
    # remove punctuation, hashes, and collapse spaces
    cleaned = re.sub(r'[^A-Za-z ]', ' ', raw)
    # extract the first TRUE or FALSE (case-insensitive)
    match = re.search(r'\b(TRUE|FALSE)\b', cleaned, re.IGNORECASE)
    if match:
        ans = match.group(1).upper()
    else:
        ans = f"Uncertain (raw: {raw.strip()})"
    return ans

with gr.Blocks(title="PhishSense 1B (LoRA) Demo") as demo:
    gr.Markdown("# 📧 PhishSense 1B — Phishing Classifier\nPaste an email and get `TRUE` (phishing) or `FALSE` (not phishing).")
    with gr.Row():
        email = gr.Textbox(
            label="Email text",
            lines=10,
            placeholder="Paste the email body here..."
        )
        result = gr.Label(label="Prediction")
    examples = gr.Examples(
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
    # Share=True lets you get a temporary public URL; remove if not needed
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
