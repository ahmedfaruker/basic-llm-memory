import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"
NUM_CTX = 512  # deliberately tiny so the context fills up fast

# Full conversation history, kept by us (the model is stateless)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]


def ask(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


turn = 0
while True:
    user_text = input("\nYou: ")
    if user_text.strip().lower() == "exit":
        break

    messages.append({"role": "user", "content": user_text})
    data = ask(messages)
    answer = data["message"]["content"]
    messages.append({"role": "assistant", "content": answer})

    turn += 1
    # Rough estimate of what we sent: ~4 characters per token
    est_tokens = sum(len(m["content"]) for m in messages) // 4

    print(f"\nModel: {answer}")
    print(
        f"[turn {turn}] messages in list: {len(messages)} | "
        f"our estimate: ~{est_tokens} tokens | "
        f"prompt_eval_count: {data.get('prompt_eval_count')} / {NUM_CTX}"
    )