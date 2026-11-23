# basic-llm-memory

A small learning project that applies the operating-system idea of virtual memory to an LLM chat: **context window = RAM**, **vector database = disk**. When the token budget fills up, old messages are moved to disk (page out), and the ones relevant to a new question are brought back (page in).

This is the first-week output of my graduation project (Context-Aware Memory Management for long-running LLM agents).

## Files

| File | What it does |
|---|---|
| `memory_manager.py` | Token budget, FIFO eviction, pinning, page out / page in with Qdrant. |
| `data/fact_test.txt` | 1 fact + 18 filler questions + 1 check question (is the fact remembered?). |
| `data/rule_test.txt` | 1 behavior rule + 18 filler questions + 1 check question (is the rule still followed?). |

## Setup

```bash
ollama pull qwen3.5:4b
ollama pull nomic-embed-text
python -m venv venv
pip install requests qdrant-client
```

Qdrant runs in **local mode** (`qdrant_data/` folder); no Docker or server needed.

## Usage

```bash
python memory_manager.py                                # interactive
python memory_manager.py data/fact_test.txt             # scripted, FIFO + recall
python memory_manager.py data/fact_test.txt --no-recall # scripted, FIFO only
python memory_manager.py data/fact_test.txt --debug     # also print top-3 scores before threshold
```

- Starting a message with `!` **pins** it: that message and the model's reply to it are never evicted.
- Type `exit` to quit.
- Each run deletes `qdrant_data/` and starts fresh, so experiments don't affect each other.

## How it works

Each turn runs these steps:

1. The user message is appended to the `messages` list.
2. **Eviction (`enforce_budget`):** while the list exceeds `LIVE_LIMIT`, the oldest non-pinned message is removed. The system message and the current question are never evicted.
3. **Page out (`page_out`):** evicted messages are embedded with `nomic-embed-text` and written to Qdrant.
4. **Page in (`page_in`):** the new question is embedded. From the `TOP_K` nearest records, those scoring above `MIN_SCORE` are taken, as many as fit in `RECALL_BUDGET`.
5. Recalled records are appended to the **system message**, and the model is called.

The budget is split like this:

```
num_ctx = 2048
├── BUDGET = 900            prompt we send
│   ├── LIVE_LIMIT = 700    messages list
│   └── RECALL_BUDGET = 200 recalled records
└── MAX_ANSWER = 250        answer (num_predict)
```

Token counts are estimated as `len(text) // 4 + 4`. The real value (`prompt_eval_count`) is printed next to it every turn.

## Experiment: FIFO vs. FIFO + recall

Each test runs 20 turns. Settings: `BUDGET=900`, `RECALL_BUDGET=200`, `MIN_SCORE=0.6`, `temperature=0`.

| Test | Mode | Rule evicted at turn | Final answer correct? | Total prompt tokens (avg/turn) |
|---|---|---|---|---|
| fact | FIFO | 5 | ✗ ("I don't know") | 10495 (524) |
| fact | FIFO + recall | 5 | ✓ (score 0.83) | 10358 (517) |
| rule | FIFO | 6 | ✗ (stopped at turn 9) | 10074 (503) |
| rule | FIFO + recall | 5 | ✗ (stopped at turn 7) | 9237 (461) |
| rule | pinned + recall | — | ✓ (20/20 turns) | 9348 (467) |

### Findings

- **Recall saves facts, not behavior rules.** "End every answer with OVER" is not semantically similar to any filler question, so it was never recalled. Only pinning kept the rule. This suggests rules need a separate layer that always stays in context, outside retrieval (the core memory vs. archival memory split in MemGPT).
- **A rule keeps working for 1–2 turns after it's evicted.** The rule itself is gone, but earlier replies ending in OVER are still in context, and the model imitates the pattern.
- **Rules are broken silently.** The model gives no sign that it forgot a rule. When it forgets a fact, it usually says "I don't know."
- **`MIN_SCORE=0.6` is too low.** All "Explain ... in detail" questions score 0.60–0.68 against each other, so irrelevant records get recalled. One of them (a 197-token reply) alone filled 98% of the recall budget. Relevant records scored 0.74 and 0.83. **Suggested value: 0.7.**
- **Pinning is cheap.** The rule plus its acknowledgment permanently take about 34 tokens.
- **The token estimate is off by up to ±20%.** It's low for short messages (chat-template overhead) and high for long English text. In practice the budget fills at about 560 real tokens.

### Known limitations

- Messages are evicted one by one, so question/answer pairs can be split.
- Long replies don't fit in `RECALL_BUDGET`, so recalled records are mostly questions without their answers.
- Questions about the conversation itself ("did I ask about X?") aren't captured well by similarity search.
- `MAX_ANSWER=250` cuts some replies mid-sentence.
- On Windows, `delete_collection` can't delete the open SQLite file and silently swallows the error. That's why the folder is deleted manually before the client opens.