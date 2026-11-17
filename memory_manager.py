import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"
NUM_CTX = 1024     # model's context window
BUDGET = 400       # max tokens we send; the rest is reserved for the answer
MAX_ANSWER = 250   # cap answer length so one reply can't flood the context


def count_tokens(msg):
    # Rough estimate: ~4 characters per token, +4 for role/template overhead
    return len(msg["content"]) // 4 + 4


def total_tokens(messages):
    return sum(count_tokens(m) for m in messages)


def enforce_budget(messages):
    """FIFO eviction: drop the oldest non-pinned messages until under budget."""
    evicted = []
    while total_tokens(messages) > BUDGET:
        # Never evict pinned messages or the newest one (the current question)
        victim = next(
            (i for i, m in enumerate(messages[:-1]) if not m["pinned"]), None
        )
        if victim is None:
            break  # nothing left that we are allowed to evict
        evicted.append(messages.pop(victim))
    return evicted


def ask(messages):
    # Send only role/content; "pinned" is our own metadata, not for the model
    payload_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    payload = {
        "model": MODEL,
        "messages": payload_messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": MAX_ANSWER},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Keep answers under 150 words.",
        "pinned": True,
    },
]

turn = 0
while True:
    user_text = input("\nYou: ").strip()
    if user_text.lower() == "exit":
        break

    # A leading "!" pins the message so it is never evicted
    pinned = user_text.startswith("!")
    if pinned:
        user_text = user_text[1:].strip()
    messages.append({"role": "user", "content": user_text, "pinned": pinned})

    for m in enforce_budget(messages):
        print(f"  [evicted] {m['role']}: {m['content'][:60]!r}")

    sent_estimate = total_tokens(messages)
    data = ask(messages)
    answer = data["message"]["content"]
    # Pin the reply too if its question was pinned (keep Q/A pairs together)
    messages.append({"role": "assistant", "content": answer, "pinned": pinned})

    turn += 1
    print(f"\nModel: {answer}")
    print(
        f"[turn {turn}] in context: {len(messages)} msgs | "
        f"our estimate: ~{sent_estimate} | "
        f"prompt_eval_count: {data.get('prompt_eval_count')} | budget: {BUDGET}"
    )