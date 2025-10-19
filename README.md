# EC601_PhishProof
This is the Repo for my EC601 Project - PhishProof. 

app_flask.py and app_gardio.py are two web apps for demo purposes. They both use the fine-tuned PhishSense model (Llama-Phishsense-1B). 
They use a prompt such as: "Classify the following text as phishing or not. Respond with 'TRUE' or 'FALSE': ...". 


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
