# Password Policy Environment

**OpenEnv Hackathon Submission** | Tag: `openenv`

---

## Overview

A black-box password generation environment where an agent must discover and satisfy
a hidden 5-rule password policy **purely through reward signal feedback**.

The agent never sees the policy rules. It submits candidate passwords and receives a
continuous score `0.0–1.0` reflecting compliance. It must infer rules by observing
how its score changes between attempts.

---

## New Features (v1.0.0)

| # | Requirement | Implementation |
|---|---|---|
| 1 | One unique password per person | `PasswordRegistry` enforces per `person_id` |
| 2 | Store successes + duplicate penalty | Registry stores hashes; duplicate → `-0.30` penalty |
| 3 | Password hashing after generation | `hashlib.sha256` — plaintext never persists |
| 4 | Parallel search with merge sort | `ThreadPoolExecutor` + merge sort + binary search |

---

## File Structure

```
my_env/
├── models.py          ← Shared types: Action, Observation, State, Reward
├── client.py          ← HTTP client (what users/inference.py imports)
├── server/
│   ├── environment.py ← All logic: policy, hashing, registry, parallel sort
│   ├── app.py         ← FastAPI server
│   └── Dockerfile     ← Container definition
├── inference.py       ← Baseline agent (OpenAI SDK + client.py)
├── openenv.yaml       ← OpenEnv manifest
├── pyproject.toml     ← Package metadata
└── requirements.txt   ← Pinned dependencies
```

---

## Hidden Policy (not visible to the agent)

| Rule | Constraint | Weight |
|------|-----------|--------|
| 1 | Length must be 5–7 characters | 0.20 |
| 2 | At least one uppercase AND one lowercase letter | 0.20 |
| 3 | At least one digit; must not be all digits | 0.20 |
| 4 | At least one symbol from `@#$%`; must not be all symbols | 0.20 |
| 5 | First character must be a letter (a–z, A–Z) or underscore `_` | 0.20 |

---

## Reward Function

```
reward = 0.20×r1 + 0.20×r2 + 0.20×r3 + 0.20×r4 + 0.20×r5
```

If a duplicate password is detected for the same person:
```
reward = max(0.0, raw_reward − 0.30)
```

### Corrected Reward Table

| Password | len | r1 | r2 | r3 | r4 | r5 | **Total** | Note |
|---|---|---|---|---|---|---|---|---|
| `hello` | 5 | 1.0 | 0.5 | 0.0 | 0.0 | 1.0 | **0.50** | no digit, no symbol |
| `HELLO123` | 8 | 0.8 | 0.5 | 1.0 | 0.0 | 1.0 | **0.66** | over length, no lower, no symbol |
| `Hello1@` | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.00** ✅ | all rules pass |
| `_Abc1@` | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.00** ✅ | all rules pass |
| `1234567` | 7 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.20** | all digits, digit start |
| `1Hello@` | 7 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | **0.80** | digit as first char |
| `abc` | 3 | 0.6 | 0.5 | 0.0 | 0.0 | 1.0 | **0.42** | too short, no digit/symbol |
| `_Abc1@` (repeat) | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **0.70** | duplicate penalty −0.30 |

---

## How Duplicate Detection Works

```
Step 1: Agent submits password "Hello1@"
Step 2: SHA-256 hash computed → stored in registry under person_id
        (parallel merge sort re-sorts the hash list)
Step 3: Agent submits "Hello1@" again
Step 4: hash("Hello1@") → binary search on sorted hash list → FOUND
Step 5: Penalty −0.30 applied → reward = 1.00 − 0.30 = 0.70
```

The registry stores only hashes. Plaintext is never persisted.

---

## Parallel Merge Sort Algorithm

```python
# Split hash list into k chunks
# Sort each chunk concurrently in ThreadPoolExecutor
# Sequential k-way merge back to one sorted list
# Binary search for O(log n) duplicate lookup
```

Time complexity: O((n/k) log(n/k)) parallel + O(n log k) merge, where k = workers.

---

## Observation Space

```json
{
  "task": "hard",
  "person_id": "agent_001",
  "attempt_number": 3,
  "steps_remaining": 7,
  "last_password": "Hello1@",
  "last_reward": 1.0,
  "best_reward_so_far": 1.0,
  "history": [
    {"step": 1, "password": "hello",   "reward": 0.50, "was_duplicate": false},
    {"step": 2, "password": "HELLO123","reward": 0.66, "was_duplicate": false},
    {"step": 3, "password": "Hello1@", "reward": 1.00, "was_duplicate": false}
  ],
  "message": "✓ Perfect score! All policy rules satisfied."
}
```

## Action Space

```json
{"person_id": "agent_001", "password": "Hello1@"}
```

---

## Tasks

### Easy — `validate_password`
Starting password: `hello` (reward = 0.50). Grader: first submission score.

### Medium — `fix_password`
Starting password: `HELLO123` (reward = 0.66). Grader: best reward (60%) + improvement (20%) + efficiency (20%).

### Hard — `generate_compliant_password`
No starting password. Grader: perfect bonus (0.50 + 0.30) + efficiency (0.20). Repeated passwords reduce score via penalty.

---

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Second terminal
export HF_TOKEN=your_token
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -f server/Dockerfile -t password-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  password-env
```

### Test manually

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{"task":"hard","person_id":"agent_001"}'
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"person_id":"agent_001","password":"_Abc1@"}'
# Expected: reward=1.0, done=true
curl http://localhost:7860/registry
```

---

## Baseline Scores

| Task | Difficulty | Expected Grade | Notes |
|------|-----------|---------------|-------|
| `validate_password` | Easy | 0.50 | Starting password score |
| `fix_password` | Medium | ~0.75 | Reaches 1.0 within 3–5 steps |
| `generate_compliant_password` | Hard | ~0.60 | Multi-constraint from scratch |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info |
| `/health` | GET | Liveness probe |
| `/reset` | POST | Start episode: `{task, person_id}` |
| `/step` | POST | Submit password: `{person_id, password}` |
| `/state` | GET | Full internal state (debug) |
| `/registry` | GET | Registry statistics |
