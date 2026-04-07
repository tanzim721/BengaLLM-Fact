import os
import time
from dotenv import load_dotenv

load_dotenv()

# ---------------- OPENAI ----------------
from openai import OpenAI

def test_openai():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "বাংলাদেশের রাজধানী কোথায়?"}
        ]
    )
    end = time.time()

    text = response.choices[0].message.content
    tokens = response.usage.total_tokens

    print("\n[OpenAI]")
    print("Response:", text)
    print("Tokens:", tokens)
    print("Latency:", round(end - start, 2), "sec")


# ---------------- GOOGLE (GEMINI) ----------------
import google.generativeai as genai

def test_google():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model = genai.GenerativeModel("gemini-1.5-pro")

    start = time.time()
    response = model.generate_content("বাংলাদেশের রাজধানী কোথায়?")
    end = time.time()

    print("\n[Google Gemini]")
    print("Response:", response.text)
    print("Latency:", round(end - start, 2), "sec")


# ---------------- TOGETHER AI ----------------
import requests

def test_together():
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/Llama-3-70b-instruct",
        "messages": [
            {"role": "user", "content": "বাংলাদেশের রাজধানী কোথায়?"}
        ]
    }

    start = time.time()
    response = requests.post(url, headers=headers, json=data)
    end = time.time()

    result = response.json()

    print("\n[Together AI - LLaMA 3]")
    print("Response:", result["choices"][0]["message"]["content"])
    print("Latency:", round(end - start, 2), "sec")


# ---------------- HUGGINGFACE ----------------
from huggingface_hub import login
from transformers import AutoTokenizer

def test_huggingface():
    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=token)

    tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")

    text = "বাংলাদেশের রাজধানী কোথায়?"
    tokens = tokenizer.tokenize(text)

    print("\n[HuggingFace - BanglaBERT]")
    print("Tokens:", tokens)


# ---------------- RUN ALL ----------------
if __name__ == "__main__":
    # test_openai()
    # test_google()
    # test_together()
    test_huggingface()