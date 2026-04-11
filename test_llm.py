"""
test_llm.py — Quick test to verify LLM is working correctly.
Run: python test_llm.py
"""

import os
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY      = os.environ.get("API_KEY", "")      # set your key here or via env var
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

if not API_KEY:
    print("[ERROR] Set API_KEY environment variable first.")
    print("  Windows: set API_KEY=your_key_here")
    print("  Linux:   export API_KEY=your_key_here")
    exit(1)

print(f"Testing LLM...")
print(f"  Model      : {MODEL_NAME}")
print(f"  API URL    : {API_BASE_URL}")
print()

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Test 1: Basic response ─────────────────────────────────────────────────────
print("=" * 50)
print("TEST 1: Basic response")
print("=" * 50)
try:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    answer = resp.choices[0].message.content.strip()
    print(f"  Response : {answer}")
    print(f"  PASSED ✓" if answer else "  FAILED ✗ (empty response)")
except Exception as e:
    print(f"  FAILED ✗ — {e}")

print()

# ── Test 2: JSON output (password agent format) ────────────────────────────────
print("=" * 50)
print("TEST 2: JSON password output (agent format)")
print("=" * 50)
try:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a password agent. Output ONLY a JSON object with "
                    "exactly two keys: 'password' and 'reasoning'. No markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current reward: 0.40. Need to add uppercase, digit, and symbol. "
                    "Last password: 'hello'. Suggest next password.\n"
                    '{"password": "...", "reasoning": "..."}'
                ),
            },
        ],
        max_tokens=100,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    print(f"  Raw response : {raw}")

    import json
    parsed = json.loads(raw)
    pw     = parsed.get("password", "")
    reason = parsed.get("reasoning", "")
    print(f"  Password     : {pw}")
    print(f"  Reasoning    : {reason}")
    print(f"  PASSED ✓" if pw else "  FAILED ✗ (empty password)")
except json.JSONDecodeError as e:
    print(f"  FAILED ✗ — JSON parse error: {e}")
    print(f"  Raw was: {raw}")
except Exception as e:
    print(f"  FAILED ✗ — {e}")

print()

# ── Test 3: Does it avoid duplicates? ─────────────────────────────────────────
print("=" * 50)
print("TEST 3: Duplicate avoidance")
print("=" * 50)
try:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a password agent. Output ONLY a JSON object with "
                    "exactly two keys: 'password' and 'reasoning'. No markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Already submitted (DO NOT repeat): ['hello', 'Ab1@xy', 'Hello1@']\n"
                    "Suggest a NEW password not in the list above.\n"
                    '{"password": "...", "reasoning": "..."}'
                ),
            },
        ],
        max_tokens=100,
        temperature=0.3,
    )
    raw    = resp.choices[0].message.content.strip()
    parsed = json.loads(raw)
    pw     = parsed.get("password", "")
    banned = ["hello", "Ab1@xy", "Hello1@"]
    if pw and pw not in banned:
        print(f"  Password : {pw} — not a duplicate ✓")
        print(f"  PASSED ✓")
    elif pw in banned:
        print(f"  Password : {pw} — IS a duplicate ✗")
        print(f"  FAILED ✗ (LLM repeated a banned password)")
    else:
        print(f"  FAILED ✗ (empty password)")
except Exception as e:
    print(f"  FAILED ✗ — {e}")

print()
print("=" * 50)
print("All tests complete.")
print("=" * 50)