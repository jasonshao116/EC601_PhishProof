import torch
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

MODEL_ID = "AcuteShrewdSecurity/Llama-Phishsense-1B"
app = Flask(__name__)

# -------- Load once at startup --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model = PeftModel.from_pretrained(base_model, MODEL_ID)
if torch.cuda.is_available():
    model = model.to("cuda")
model.eval()

HTML = """
<!doctype html>
<title>PhishSense 1B — Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; height: 240px; }
  .result { margin-top: 1rem; font-weight: 700; }
  button { padding: .6rem 1rem; font-size: 1rem; }
</style>
<h1>📧 PhishSense 1B — Phishing Classifier</h1>
<form method="POST" action="/predict">
  <textarea name="email" placeholder="Paste email text here...">{{ email|default("") }}</textarea>
  <div><button type="submit">Classify</button></div>
</form>
{% if pred is not none %}
  <div class="result">Prediction: {{ pred }}</div>
{% endif %}
"""

def classify(email_text: str) -> str:
    prompt = (
        "Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE':\n\n"
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
    # --- robust parsing: find the first TRUE/FALSE after "Answer:" ---
    raw = text.split("Answer:")[-1].strip()
    cleaned = re.sub(r'[^A-Za-z ]', ' ', raw)          # drop punctuation/# etc.
    m = re.search(r'\b(TRUE|FALSE)\b', cleaned, re.I)  # pick the first valid label
    if m:
        return m.group(1).upper()
    else:
        return f"Uncertain (raw: {raw})"

@app.get("/")
def index():
    return render_template_string(HTML, pred=None)

@app.post("/predict")
def predict():
    email = request.form.get("email", "")
    pred = classify(email) if email.strip() else "Please paste an email."
    return render_template_string(HTML, pred=pred, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
