"""
inference.py — Adaptive agent for the Password Policy Environment.

Phase 1: Systematic feature probing (one new feature per step)
Phase 2: Reward-guided adaptation (LLM synthesises best candidate + mutates)

Environment variables:
  API_BASE_URL   — LLM API endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME     — model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       — Hugging Face bearer token
  ENV_BASE_URL   — Password env server URL (default: http://localhost:7860)

stdout format (mandatory — zero deviation):
  [START] task=<name> env=password-policy-env model=<model>
  [STEP]  step=<n> action=<password> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import importlib.util

# ── Fix import paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR  = os.path.join(BASE_DIR, "src", "envs", "my_env")

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

models_mod = _load_module("models", os.path.join(ENV_DIR, "models.py"))
client_mod = _load_module("client", os.path.join(ENV_DIR, "client.py"))

PasswordEnvClient = client_mod.PasswordEnvClient
Action            = models_mod.Action
Observation       = models_mod.Observation
State             = models_mod.State
Reward            = models_mod.Reward

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://andrew2712-my-env.hf.space")

TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = 10

# ── OpenAI client ──────────────────────────────────────────────────────────────

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Phase 1: systematic feature probe sequence ─────────────────────────────────
# Each step adds exactly ONE new feature so reward deltas are interpretable.
PROBE_SEQUENCE = [
    ("abc",           "baseline: short, lowercase only"),
    ("abcdef",        "+length: 6 chars"),
    ("Abcdef",        "+uppercase: capital first letter"),
    ("Abcdef1",       "+digit: one digit appended"),
    ("Abcdef1@",      "+symbol: one symbol appended"),
    ("_Abcdef1@",     "+underscore start: leading underscore"),
    ("_Abcde1@",      "length variant: 8 chars with underscore"),
    ("_Abc1@X",       "variation: symbol mid, extra upper"),
]

# ── System prompt for Phase 2 (adaptive) ──────────────────────────────────────

ADAPTIVE_SYSTEM_PROMPT = """You are an expert password generation agent in a \
black-box optimisation environment.

TASK:
Generate a password satisfying a completely hidden policy. After each attempt \
you receive a reward score between 0.0 and 1.0. A score of 1.0 means all rules \
are satisfied. You are NEVER told what the rules are.

IMPORTANT RULES:
- Never repeat a password you have already submitted. Repeated passwords incur \
a -0.30 penalty.
- Each password must be a fresh, unique attempt.

STRATEGY — reward delta analysis:
You will be given the full attempt history with reward scores.
1. Find the step where reward jumped the most (largest positive delta).
2. Identify WHICH single feature changed at that step — that feature is likely \
required by the policy.
3. Take the current best password and make ONE small targeted mutation to test \
a new hypothesis (e.g. change length by 1, swap symbol, change case pattern).
4. Never undo a feature that produced a positive delta.
5. If reward is stuck, try: different symbol (@#$%), different length (5/6/7/8), \
digit at start vs end, all-lowercase vs mixed case.

OUTPUT FORMAT:
Respond ONLY with a JSON object — no markdown, no explanation:
{"password": "YourPasswordHere", "reasoning": "one line rationale"}
"""


def _build_adaptive_prompt(obs: dict, step: int) -> str:
    """Build the user prompt for Phase 2 adaptive reasoning."""
    history = obs.get("history", [])
    lines   = []
    for i, r in enumerate(history):
        prev_reward = history[i - 1]["reward"] if i > 0 else 0.0
        delta       = r["reward"] - prev_reward
        delta_str   = f"Δ{delta:+.2f}" if i > 0 else "     "
        dup         = " [DUPLICATE -0.30]" if r.get("was_duplicate") else ""
        lines.append(
            f"  step={r['step']:>2}  reward={r['reward']:.2f}  {delta_str}"
            f"  password='{r['password']}'{dup}"
        )
    history_str = "\n".join(lines) if lines else "  (no attempts yet)"

    return (
        f"Step {step} of {MAX_STEPS}\n"
        f"Best reward so far : {obs.get('best_reward_so_far', 0.0):.2f}\n"
        f"Steps remaining    : {obs.get('steps_remaining', 0)}\n\n"
        f"Full attempt history (with reward deltas):\n{history_str}\n\n"
        "Analyse the deltas, identify which features help, and produce your "
        "next password. Respond ONLY with the JSON object."
    )


# ── Agent decision logic ───────────────────────────────────────────────────────

def get_agent_action(obs: dict, step: int, submitted: set) -> str:
    """
    Phase 1 (steps 1-8): return next probe password from PROBE_SEQUENCE.
    Phase 2 (steps 9+):  call LLM with full delta-annotated history.
    Falls back to a safe unique candidate if LLM fails or repeats.
    """
    # ── Phase 1: systematic probe ──────────────────────────────────────────────
    probe_idx = step - 1
    if probe_idx < len(PROBE_SEQUENCE):
        candidate, _ = PROBE_SEQUENCE[probe_idx]
        if candidate not in submitted:
            return candidate
        # If somehow already submitted (e.g. episode reuse), fall through

    # ── Phase 2: LLM adaptive reasoning ───────────────────────────────────────
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": ADAPTIVE_SYSTEM_PROMPT},
                {"role": "user",   "content": _build_adaptive_prompt(obs, step)},
            ],
            max_tokens=256,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        parsed = json.loads(raw)
        pw = str(parsed.get("password", "")).strip()
        if pw and pw not in submitted:
            return pw
        # LLM returned a duplicate — fall through to safe fallback
    except Exception:
        pass

    # ── Safe unique fallback ───────────────────────────────────────────────────
    # Mutate the best-seen password by tweaking length and symbol.
    best_pw    = obs.get("last_password", "_Abc1@") or "_Abc1@"
    symbols    = ["@", "#", "$", "%"]
    base_cores = [
        f"_Ab{step}@X", f"_Bc{step}#Y", f"_Cd{step}$Z",
        f"_De{step}%W", f"_Ef{step}@V", f"_Fg{step}#U",
    ]
    for core in base_cores:
        if core not in submitted:
            return core
    # Last resort: suffix the step number to the best password
    for sym in symbols:
        candidate = f"{best_pw[:5]}{step}{sym}"
        if candidate not in submitted:
            return candidate

    return f"_Z{step}a1@"


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task: str, person_id: str) -> None:
    rewards:     list[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    error_msg:   str         = "null"
    submitted:   set         = set()   # track submitted passwords this episode

    print(
        f"[START] task={task} env=password-policy-env model={MODEL_NAME}",
        flush=True,
    )

    with PasswordEnvClient(base_url=ENV_BASE_URL) as env:
        try:
            obs_obj = env.reset(task=task, person_id=person_id)
            obs     = obs_obj.model_dump()

            for step in range(1, MAX_STEPS + 1):
                try:
                    password  = get_agent_action(obs, step, submitted)
                    submitted.add(password)
                    error_msg = "null"
                except Exception as e:
                    password  = f"_Fb{step}@1"
                    submitted.add(password)
                    error_msg = str(e).replace("\n", " ")

                try:
                    obs_obj, reward, done, info = env.step(
                        person_id=person_id,
                        password=password,
                    )
                    obs       = obs_obj.model_dump()
                    error_msg = "null"
                except Exception as e:
                    reward    = 0.0
                    done      = True
                    error_msg = str(e).replace("\n", " ")

                rewards.append(reward)
                steps_taken = step
                done_str    = "true" if done else "false"

                print(
                    f"[STEP] step={step} action={password} "
                    f"reward={reward:.2f} done={done_str} error={error_msg}",
                    flush=True,
                )

                if done:
                    success = (reward >= 1.0)
                    break

        except Exception as e:
            error_msg = str(e).replace("\n", " ")
            success   = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={steps_taken} rewards={rewards_str}",
        flush=True,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task in TASKS:
        run_episode(task=task, person_id=f"adaptive_agent_{task}")
        print(flush=True)
