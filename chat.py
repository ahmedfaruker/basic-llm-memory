import time
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"
NS = 1e9  # Ollama reports durations in nanoseconds

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
        {"role": "user", "content": "What is virtual memory in one sentence?"},
    ],
    "stream": False,                  # return the full answer at once
    "think": False,                   # disable hidden reasoning tokens
    "options": {"temperature": 0},    # deterministic output for experiments
}

start = time.perf_counter()
resp = requests.post(OLLAMA_URL, json=payload)
wall = time.perf_counter() - start
resp.raise_for_status()  # crash loudly on HTTP errors
data = resp.json()

print("=== ANSWER ===")
print(data["message"]["content"])

print("=== METRICS ===")
print(f"input tokens  (prompt_eval_count): {data.get('prompt_eval_count')}")
print(f"output tokens (eval_count):        {data.get('eval_count')}")
print(f"load time:      {data['load_duration'] / NS:.2f} s")
print(f"prompt time:    {data.get('prompt_eval_duration', 0) / NS:.2f} s")
print(f"generate time:  {data['eval_duration'] / NS:.2f} s")
print(f"total (Ollama): {data['total_duration'] / NS:.2f} s")
print(f"total (wall):   {wall:.2f} s")