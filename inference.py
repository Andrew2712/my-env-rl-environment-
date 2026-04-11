"""
inference.py — Baseline agent for the Password Policy Environment.

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

# ── Fix import paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR  = os.path.join(BASE_DIR, "src", "envs", "my_env")
sys.path.insert(0, ENV_DIR)

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = 10

# ── Lazy OpenAI client ─────────────────────────────────────────────────────────

_llm = None

def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    try:
        from openai import OpenAI
        token = HF_TOKEN if HF_TOKEN else "dummy-token"
        _llm  = OpenAI(base_url=API_BASE_URL, api_key=token)
    except Exception as e:
        print(f"[WARN] LLM init failed: {e}", flush=True)
        _llm = None
    return _llm

def get_env_client():
    try:
        from client import PasswordEnvClient
        return PasswordEnvClient
    except Exception as e:
        raise RuntimeError(f"Could not import PasswordEnvClient: {e}")

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a password policy inference agent. Your job is to \
discover a hidden password policy through trial and error using reward feedback.

REWARD: A score 0.0-1.0. Each of 5 rules contributes 0.20.
Reward 1.0 = all rules satisfied. You must reach 1.0.

THE 5 HIDDEN RULES (you must discover these from rewards):
Each rule is worth exactly 0.20. When you satisfy a new rule, reward goes up by 0.20.

REASONING PROTOCOL — follow this exactly:
1. Compare current reward to previous reward.
2. If reward INCREASED by 0.20: your last change fixed one rule. KEEP that change.
3. If reward UNCHANGED: your last change had no effect. Try a different dimension.
4. If reward DECREASED: your last change broke a rule. REVERT it immediately.
5. If reward = 0.80: exactly ONE rule still failing. Focus only on that.
6. If reward = 1.00: done!

DIMENSIONS TO EXPLORE (change ONE at a time):
- Length: try 5, 6, 7 characters
- Uppercase: include A-Z
- Lowercase: include a-z
- Digit: include 0-9
- Symbol: try @, then #, then $, then %  (ONLY these 4 are valid)
- First char: try starting with a letter or underscore _

NEVER:
- Repeat a password (penalty -0.30)
- Change more than one thing at a time
- Use symbols outside @#$%

OUTPUT — JSON only, no markdown:
{"password": "...", "reasoning": "what changed and why"}
"""

def build_prompt(obs: dict, step: int, episode_history: list) -> str:
    """Build a rich prompt showing the full reward trajectory."""
    lines = []
    for i, h in enumerate(episode_history):
        prev_r = episode_history[i-1]["reward"] if i > 0 else None
        if prev_r is not None:
            delta = h["reward"] - prev_r
            if delta > 0.001:
                trend = f"  UP +{delta:.2f} <- last change HELPED"
            elif delta < -0.001:
                trend = f"  DOWN {delta:.2f} <- last change HURT"
            else:
                trend = "  SAME <- no effect"
        else:
            trend = "  (first attempt)"
        dup = " [DUPLICATE -0.30]" if h.get("was_duplicate") else ""
        lines.append(
            f"  step={h['step']:2d}  pw='{h['password']}'  "
            f"reward={h['reward']:.2f}{trend}{dup}"
        )

    history_str = "\n".join(lines) if lines else "  (none yet)"

    best   = obs.get("best_reward_so_far", 0.0)
    last_r = obs.get("last_reward", 0.0)
    left   = obs.get("steps_remaining", 0)
    last_pw = obs.get("last_password", "")

    # Smart contextual hint based on current reward
    rules_done   = round(last_r / 0.2) if last_r > 0 else 0
    rules_left   = 5 - rules_done
    if last_r >= 1.0:
        hint = "Already perfect! (should be done)"
    elif last_r == 0.8:
        hint = "ONE rule left. Most likely: try adding symbol @, then #, then $, then %. Or check first character is letter/_."
    elif last_r == 0.6:
        hint = "TWO rules left. Fix one dimension at a time."
    elif last_r == 0.4:
        hint = "THREE rules left. Start with a broader password covering more dimensions."
    elif last_r == 0.0:
        hint = "No rules passing yet. Use a diverse starting password."
    else:
        hint = f"{rules_left} rules left. Keep isolating changes."

    return (
        f"=== Step {step}/{MAX_STEPS} | Task | Steps left: {left} ===\n\n"
        f"Last password : '{last_pw}'\n"
        f"Last reward   : {last_r:.2f}  |  Best so far: {best:.2f}\n"
        f"Hint          : {hint}\n\n"
        f"Full reward trajectory:\n{history_str}\n\n"
        f"Rules satisfied so far: {rules_done}/5 (each worth 0.20)\n\n"
        "Decide your next password. Make ONE targeted change. Output JSON only."
    )

def get_agent_action(obs: dict, step: int, episode_history: list) -> str:
    """Call LLM with full trajectory context, fall back to rule-based search."""
    llm = get_llm()

    if llm is not None:
        try:
            resp = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(obs, step, episode_history)},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            parsed = json.loads(raw)
            pw = str(parsed.get("password", "")).strip()
            if pw and pw not in [h["password"] for h in episode_history]:
                return pw
        except Exception:
            pass

    # ── Rule-based fallback: systematic search grid ────────────────────────────
    # Each candidate tests a specific rule dimension.
    # Ordered from most-likely-to-succeed to least.
    submitted = {h["password"] for h in episode_history}
    candidates = [
        # (password, what it tests)
        ("Ab1@xy",  "all rules: upper+lower+digit+@+letter-start+len6"),
        ("Ab1#xy",  "symbol # instead of @"),
        ("Ab1$xy",  "symbol $ instead of @"),
        ("Ab1%xy",  "symbol % instead of @"),
        ("_Ab1@x",  "underscore first char"),
        ("Ab1@x",   "length 5"),
        ("Ab1@xyz",  "length 7"),
        ("Bc2@yz",  "different base, symbol @"),
        ("Cd3#za",  "different base, symbol #"),
        ("De4$ab",  "different base, symbol $"),
    ]
    for pw, _ in candidates:
        if pw not in submitted:
            return pw

    # Last resort: generate unique variant
    import random, string
    while True:
        pw = (
            random.choice(string.ascii_uppercase) +
            random.choice(string.ascii_lowercase) +
            str(random.randint(1, 9)) +
            random.choice("@#$%") +
            "".join(random.choices(string.ascii_lowercase, k=2))
        )
        if pw not in submitted:
            return pw

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task: str, person_id: str) -> None:
    rewards:          list[float] = []
    steps_taken:      int         = 0
    success:          bool        = False
    error_msg:        str         = "null"
    episode_history:  list        = []   # track full trajectory for LLM context

    print(
        f"[START] task={task} env=password-policy-env model={MODEL_NAME}",
        flush=True,
    )

    try:
        PasswordEnvClient = get_env_client()
    except RuntimeError as e:
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    try:
        with PasswordEnvClient(base_url=ENV_BASE_URL) as env:
            try:
                obs_obj = env.reset(task=task, person_id=person_id)
                obs     = obs_obj.model_dump()

                for step in range(1, MAX_STEPS + 1):
                    # Agent picks action using full episode history
                    try:
                        password  = get_agent_action(obs, step, episode_history)
                        error_msg = "null"
                    except Exception as e:
                        password  = "Ab1@xy"
                        error_msg = str(e).replace("\n", " ")

                    # Environment step
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

                    # Record in episode history for LLM context
                    was_dup = (
                        obs.get("history", [{}])[-1].get("was_duplicate", False)
                        if obs.get("history") else False
                    )
                    episode_history.append({
                        "step":          step,
                        "password":      password,
                        "reward":        reward,
                        "was_duplicate": was_dup,
                    })

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
        run_episode(task=task, person_id=f"baseline_agent_{task}")
        print(flush=True)
