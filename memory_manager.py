import uuid

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import os
import shutil

OLLAMA_URL = "http://localhost:11434/api/chat"
EMBED_URL = "http://localhost:11434/api/embed"
MODEL = "qwen3.5:4b"
EMBED_MODEL = "nomic-embed-text"

NUM_CTX = 2048       # model's context window
BUDGET = 900         # max tokens we send; the rest is reserved for the answer
MAX_ANSWER = 250     # cap answer length so one reply can't flood the context
RECALL_BUDGET = 200  # tokens reserved inside BUDGET for paged-in messages
TOP_K = 3            # how many records to fetch from the vector store
USE_RECALL = True    # False = plain FIFO (for the comparison experiment)
LIVE_LIMIT = BUDGET - RECALL_BUDGET if USE_RECALL else BUDGET  # cap for the messages list
MIN_SCORE = 0.6   # constants section

COLLECTION = "paged_out"
QDRANT_PATH = "qdrant_data"
# Wipe the store before the client opens any file (Windows can't delete open files)
if os.path.exists(QDRANT_PATH):
    shutil.rmtree(QDRANT_PATH)  # no ignore_errors: fail loudly if it's locked
qdrant = QdrantClient(path=QDRANT_PATH)


# ---------- token accounting ----------

def count_tokens(msg):
    # Rough estimate: ~4 characters per token, +4 for role/template overhead
    return len(msg["content"]) // 4 + 4


def total_tokens(messages):
    return sum(count_tokens(m) for m in messages)

def bar(used, limit, width=20):
    filled = min(width, round(width * used / limit)) if limit else 0
    return "█" * filled + "░" * (width - filled)


def show_budget(live, recall, real):
    rows = [
        ("live", live, LIVE_LIMIT),
        ("recall", recall, RECALL_BUDGET),
        ("total", live + recall, BUDGET),
    ]
    for name, used, limit in rows:
        print(f"  {name:<6} [{bar(used, limit)}] {used:>4}/{limit:<4} ({used / limit:4.0%})")
    print(f"  real prompt tokens (prompt_eval_count): {real}")

def enforce_budget(messages):
    """FIFO eviction: drop the oldest non-pinned messages until under budget."""
    evicted = []
    while total_tokens(messages) > LIVE_LIMIT:
        # Never evict pinned messages or the newest one (the current question)
        victim = next(
            (i for i, m in enumerate(messages[:-1]) if not m["pinned"]), None
        )
        if victim is None:
            break  # nothing left that we are allowed to evict
        evicted.append(messages.pop(victim))
    return evicted


# ---------- vector store (the "disk") ----------

def embed(text, kind):
    # nomic-embed-text expects a task prefix: "search_document" or "search_query"
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": f"{kind}: {text}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def init_store():
    dim = len(embed("probe", "search_document"))
    qdrant.create_collection(
        COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )


def page_out(evicted):
    """Write evicted messages to the vector store instead of dropping them."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(m["content"], "search_document"),
            payload={"role": m["role"], "content": m["content"]},
        )
        for m in evicted
    ]
    if points:
        qdrant.upsert(COLLECTION, points=points)


def page_in(query):
    """Fetch the most similar paged-out messages that fit in RECALL_BUDGET."""
    hits = qdrant.query_points(
        COLLECTION, query=embed(query, "search_query"), limit=TOP_K,
        score_threshold=MIN_SCORE,
    ).points
    recalled, used = [], 0
    for h in hits:
        cost = count_tokens(h.payload)
        if used + cost > RECALL_BUDGET:
            continue
        recalled.append({**h.payload, "score": h.score})
        used += cost
    return recalled


# ---------- model call ----------

def ask(messages, recalled):
    # Send only role/content; "pinned" is our own metadata, not for the model
    payload_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    if recalled:
        # Merge recalled memories into the system prompt (safest across chat templates)
        speaker = {"user": "The user said", "assistant": "You (the assistant) said"}
        notes = "\n".join(f"- {speaker[r['role']]}: {r['content']}" for r in recalled)
        payload_messages[0]["content"] += (
            f"\n\nRelevant earlier messages (from long-term memory):\n{notes}"
        )

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


# ---------- main loop ----------

def main():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep answers under 150 words.",
            "pinned": True,
        },
    ]
    init_store()

    turn = 0
    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() == "exit":
            break
        if not user_text:
            continue

        # A leading "!" pins the message so it is never evicted
        pinned = user_text.startswith("!")
        if pinned:
            user_text = user_text[1:].strip()
        messages.append({"role": "user", "content": user_text, "pinned": pinned})

        evicted = enforce_budget(messages)
        for m in evicted:
            print(f"  [evicted] {m['role']}: {m['content'][:60]!r}")
        page_out(evicted)

        recalled = page_in(user_text) if USE_RECALL else []
        for r in recalled:
            print(f"  [paged in] {r['score']:.2f} {r['role']}: {r['content'][:60]!r}")

        live = total_tokens(messages)
        recall = sum(count_tokens(r) for r in recalled)
        data = ask(messages, recalled)
        answer = data["message"]["content"]
        # Pin the reply too if its question was pinned (keep Q/A pairs together)
        messages.append({"role": "assistant", "content": answer, "pinned": pinned})

        turn += 1
        print(f"\nModel: {answer}")
        print(f"[turn {turn}] in context: {len(messages)} msgs | recalled: {len(recalled)}")
        show_budget(live, recall, data.get("prompt_eval_count"))


if __name__ == "__main__":
    try:
        main()
    finally:
        qdrant.close()  # close explicitly to avoid the noisy error at interpreter shutdown