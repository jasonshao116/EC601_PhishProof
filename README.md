# EC601_PhishProof

This is the Repo for my EC601 Project - PhishProof. 

app_flask.py and app_gardio.py are two web apps for demo purposes. They both use the fine-tuned PhishSense model (Llama-Phishsense-1B). 
They use a prompt such as: "Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE': ...". 


### **train_lora.py**

What it does: 
- Fine-tunes a base LLM with LoRA (PEFT) to perform binary phishing classification.
- Reads a CSV (expects message_content and is_spam columns), builds prompts like: “Classify the following text as phishing or not… Answer: TRUE/FALSE”, then tokenizes and trains.
- Supports optional 4-bit loading (bitsandbytes) and is friendly to Apple-silicon (MPS) setups.

Model & Training Data: 
- Loads any causal LLM via AutoModelForCausalLM.from_pretrained(args.base_model) with the matching AutoTokenizer. 
- We use meta-llama/Llama-3.2-1B-Instruct (https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) as the base model. 
- spam_dataset.csv contains our training dataset. It includes 1,000 entries of spam and non-spam messages with realistic content. Source: https://www.kaggle.com/datasets/devildyno/email-spam-or-not-classification 

Outputs:
- Saves only the LoRA adapter + tokenizer to --output_dir (not the full merged model). 
- We use this adapter-augmented model to run the TRUE/FALSE phishing classifier via generation.

How to run it: 
```bash
pip install -U datasets
python train_lora.py --csv_path Training_data/spam_dataset.csv --output_dir phishsense_lora_adapter --hf_token $HF_TOKEN --max_length 512 --batch_size 1 --eval_batch_size 1 --gradient_accumulation_steps 16 
# Replace $HF_TOKEN with your Hugging Face Token to gain access to meta-llama/Llama-3.2-1B-Instruct model. 
# This is just an example usage. The parameters after $HF_TOKEN can be changed to adjust speed, memory usage, and training quality.
```


### **app_gradio.py — Quick interactive demo (no HTML required)**

What it does:
- Uses Gradio, a Python library that automatically creates a web UI around your function.
- You only define the function predict_single(email_text) and Gradio handles:
  - The text input box
  - The “Classify” button
  - The output label
  - Hosting the app at http://localhost:7860

Benefits: 
- No frontend code needed; one line builds the interface.
- Great for ML demos, notebooks, or quick sharing.
- Optional “share” link lets you send a public URL (gradio.live) to others instantly.
- Live reload and queues for concurrent requests are built in.

Example usage: 
- You’re testing your LLM.
- You want to show a demo in a meeting or class.
- You want to share an interactive prototype. 

How to run it: 
```bash
pip install torch transformers peft gradio
python app_gradio.py
```
On your local machine, go to http://localhost:7860 


### **app_flask.py — Traditional web server (full control)**

What it does:
- Uses Flask, a lightweight Python web framework.
- You manually define:
  - The HTML form (textarea + button)
  - Routes (/ and /predict) 
  - Rendering logic using Jinja2 templates

Benefits: 
- Full control of design, routing, and integration.
- You can add authentication, REST APIs, file uploads, databases, etc.
- More suitable for production or deployment on a real web server (like AWS, GCP, etc.).

Example usage: 
- You want to integrate phishing detection into a bigger web system (dashboard, analytics).
- You’re building a backend endpoint /predict to serve model results to another service.

How to run it: 
```bash
pip install torch transformers peft flask
python app_flask.py
```
On your local machine, go to http://localhost:5050
