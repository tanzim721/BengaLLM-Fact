import os
import time
import csv
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "results/api_latency_log.csv"

# Create CSV if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["provider", "model", "response", "tokens", "latency_sec"])


def log_result(provider, model, response, tokens, latency):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([provider, model, response, tokens, round(latency, 3)])


# ---------------- OPENAI ----------------
from openai import OpenAI

def test_openai():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "বাংলায় উত্তর দাও: বাংলাদেশের রাজধানী কোথায়?"

    start = time.time()
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    end = time.time()

    text = res.choices[0].message.content
    tokens = res.usage.total_tokens

    print("\n[OpenAI]")
    print(text)

    log_result("OpenAI", "gpt-4o", text, tokens, end - start)


# ---------------- GOOGLE ----------------
import google.generativeai as genai

def test_google():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = "বাংলায় উত্তর দাও: বাংলাদেশের রাজধানী কোথায়?"

    start = time.time()
    res = model.generate_content(prompt)
    end = time.time()

    text = res.text

    print("\n[Google Gemini]")
    print(text)

    log_result("Google", "gemini-1.5-pro", text, "-", end - start)


# ---------------- TOGETHER ----------------
import requests

def test_together():
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }

    prompt = "বাংলায় উত্তর দাও: বাংলাদেশের রাজধানী কোথায়?"

    data = {
        "model": "meta-llama/Llama-3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    start = time.time()
    res = requests.post(url, headers=headers, json=data)
    end = time.time()

    result = res.json()
    text = result["choices"][0]["message"]["content"]

    print("\n[Together AI]")
    print(text)

    log_result("Together", "LLaMA-3-70B", text, "-", end - start)


# ---------------- HUGGINGFACE ----------------
from huggingface_hub import login
from transformers import AutoTokenizer

def test_huggingface():
    token = os.getenv("HF_TOKEN")
    login(token=token)

    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")

    text = "বাংলাদেশের রাজধানী কোথায়?"
    tokens = tokenizer.tokenize(text)

    print("\n[HuggingFace Tokenizer]")
    print(tokens)

    log_result("HuggingFace", "BanglaBERT-tokenizer", str(tokens), len(tokens), 0)


# ---------------- RUN ----------------
if __name__ == "__main__":
    # test_openai()
    # test_google()
    # test_together()
    test_huggingface()