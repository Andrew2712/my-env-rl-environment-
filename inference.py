"""
inference.py — Agent for the Password Policy Environment.

Environment variables (injected by hackathon proxy):
  API_BASE_URL   — LLM API endpoint (injected by hackathon)
  API_KEY        — API key (injected by hackathon)
  MODEL_NAME     — model identifier
  ENV_BASE_URL   — Password env server URL (default: http://localhost:7860)

stdout format (mandatory — zero deviation):
  [START] task=<n> env=password-policy-env model=<model>
  [STEP]  step=<n> action=<password> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import random
import string
import re

# ── Fix import paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(BASE_DIR, "src", "envs", "my_env")
sys.path.insert(0, ENV_DIR)

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 10

if not API_BASE_URL or not API_KEY:
    print("[ERROR] API_BASE_URL or API_KEY environment variable is missing!", flush=True)

# ── OpenAI client for hackathon proxy ────────────────────────────────────────
def make_llm():
    if not API_BASE_URL or not API_KEY:
        print("[WARN] LLM disabled - missing API_BASE_URL or API_KEY", flush=True)
        return None

    try:
        from openai import OpenAI

        base_url = API_BASE_URL.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        client = OpenAI(
            base_url=base_url,
            api_key=API_KEY,
            timeout=60.0,
            max_retries=2,
        )

        print(f"[INFO] LLM client initialized successfully | base_url={base_url} | model={MODEL_NAME}", flush=True)
        return client
    except Exception as e:
        print(f"[ERROR] Failed to create OpenAI client: {e}", flush=True)
        return None


def get_env_client():
    try:
        from client import PasswordEnvClient
        return PasswordEnvClient
    except Exception as e:
        raise RuntimeError(f"Could not import PasswordEnvClient: {e}")


# ── System Prompt (unchanged) ────────────────────────────────────────────────
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
- Length: try 5, 6, 7, 8 characters
- Uppercase: include A-Z
- Lowercase: include a-z
- Digit: include 0-9
- Symbol: try @, then #, then $, then % (ONLY these 4 are valid)
- First char: try starting with a letter or underscore _
NEVER:
- Repeat a password (penalty -0.30)
- Change more than one thing at a time
- Use symbols outside @#$%
CRITICAL OUTPUT RULES:
- Output ONLY a JSON object. No markdown. No explanation outside the JSON.
- The JSON must have exactly two keys: "password" and "reasoning"
- Example: {"password": "Ab1@xy", "reasoning": "Added digit to test digit rule"}
"""


def build_prompt(obs: dict, step: int, episode_history: list) -> str:
    """Build a rich prompt showing the full reward trajectory."""
    lines = []
    for i, h in enumerate(episode_history):
        prev_r = episode_history[i-1]["reward"] if i > 0 else None
        if prev_r is not None:
            delta = h["reward"] - prev_r
            if delta > 0.001:
                trend = f" UP +{delta:.2f} <- last change HELPED"
            elif delta < -0.001:
                trend = f" DOWN {delta:.2f} <- last change HURT"
            else:
                trend = " SAME <- no effect"
        else:
            trend = " (first attempt)"
        dup = " [DUPLICATE -0.30]" if h.get("was_duplicate") else ""
        lines.append(
            f" step={h['step']:2d} pw='{h['password']}' "
            f"reward={h['reward']:.2f}{trend}{dup}"
        )
    history_str = "\n".join(lines) if lines else " (none yet)"
    best = obs.get("best_reward_so_far", 0.0)
    last_r = obs.get("last_reward", 0.0)
    left = obs.get("steps_remaining", 0)
    last_pw = obs.get("last_password", "")
    best_entry = max(episode_history, key=lambda h: h["reward"]) if episode_history else None
    best_pw_info = (
        f"Best password so far: '{best_entry['password']}' (reward={best_entry['reward']:.2f})"
        if best_entry else "No attempts yet"
    )
    rules_done = round(last_r / 0.2) if last_r > 0 else 0
    rules_left = 5 - rules_done
    if last_r >= 1.0:
        hint = "Already perfect! (should be done)"
    elif last_r == 0.8:
        hint = "ONE rule left. Try changing ONE thing: symbol (@#$%), length, or first character."
    elif last_r == 0.6:
        hint = "TWO rules left. Fix one dimension at a time. Start from your best password."
    elif last_r == 0.4:
        hint = "THREE rules left. Start with a broader password covering more dimensions."
    elif last_r == 0.0:
        hint = "No rules passing yet. Use a diverse starting password like Ab1@xy."
    else:
        hint = f"{rules_left} rules left. Keep isolating changes. Always start from your best password."
    dup_warning = obs.get("_duplicate_warning", "")
    dup_section = f"\nWARNING: {dup_warning}\n" if dup_warning else ""
    submitted_list = [h["password"] for h in episode_history]
    return (
        f"=== Step {step}/{MAX_STEPS} | Steps left: {left} ===\n\n"
        f"Last password : '{last_pw}'\n"
        f"Last reward : {last_r:.2f} | Best so far: {best:.2f}\n"
        f"{best_pw_info}\n"
        f"Hint : {hint}\n"
        f"{dup_section}\n"
        f"Full reward trajectory:\n{history_str}\n\n"
        f"Rules satisfied so far: {rules_done}/5 (each worth 0.20)\n\n"
        f"Already submitted (DO NOT repeat these): {submitted_list}\n\n"
        "Decide your next password. Make ONE targeted change from your best password. "
        "Output a JSON object only — no markdown, no extra text:\n"
        '{"password": "...", "reasoning": "..."}'
    )


def clean_llm_response(raw: str) -> dict:
    """Robustly extract JSON from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        for part in parts[1:]:
            cleaned = part.lstrip("json").strip()
            if cleaned:
                raw = cleaned
                break
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        raw = raw[brace_start : brace_end + 1]
    return json.loads(raw)


def get_agent_action(obs: dict, step: int, episode_history: list, llm) -> tuple[str, str]:
    submitted = {h["password"] for h in episode_history}
    current_obs = obs

    # ── LLM path (up to 3 retries) ────────────────────────────────────────────
    if llm is not None:
        last_llm_error = None
        for attempt in range(3):
            try:
                prompt = build_prompt(current_obs, step, episode_history)

                resp = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.3 + attempt * 0.1,
                    timeout=45.0,
                )

                raw = resp.choices[0].message.content.strip()
                parsed = clean_llm_response(raw)
                pw = str(parsed.get("password", "")).strip()

                if not pw:
                    last_llm_error = "LLM returned empty password"
                    print(f"[WARN] attempt={attempt+1} empty password from LLM", flush=True)
                    continue

                if pw in submitted:
                    last_llm_error = f"LLM suggested duplicate: {pw}"
                    print(f"[WARN] attempt={attempt+1} duplicate suggested, retrying", flush=True)
                    current_obs = dict(current_obs)
                    current_obs["_duplicate_warning"] = f"'{pw}' was already tried — pick a completely DIFFERENT password."
                    continue

                print(f"[INFO] LLM suggested password: {pw}", flush=True)
                return pw, "llm"

            except Exception as e:
                last_llm_error = f"LLM call error: {e}"
                print(f"[WARN] LLM attempt {attempt+1} failed: {e}", flush=True)

        print(f"[WARN] All LLM attempts failed. Last error: {last_llm_error}. Using fallback.", flush=True)

    # ── Smart fallback (your original logic kept) ─────────────────────────────
    best_entry = max(episode_history, key=lambda h: h["reward"]) if episode_history else None
    best_pw = best_entry["password"] if best_entry else "Ab1@xy"
    symbols = ["@", "#", "$", "%"]
    lengths = [5, 6, 7, 8]
    fallback_candidates = []

    for sym in symbols:
        mutated = re.sub(r"[@#$%]", sym, best_pw)
        fallback_candidates.append(mutated if mutated != best_pw else best_pw + sym)

    for length in lengths:
        if len(best_pw) < length:
            fallback_candidates.append(best_pw + "a" * (length - len(best_pw)))
        elif len(best_pw) > length:
            fallback_candidates.append(best_pw[:length])

    fallback_candidates.append("_" + best_pw)
    fallback_candidates.append("A" + best_pw[1:] if len(best_pw) > 1 else "A" + best_pw)

    fallback_candidates += [
        "Ab1@xy", "Ab1#xy", "Ab1$xy", "Ab1%xy",
        "_Ab1@x", "Ab1@x", "Ab1@xyz", "Ab1@xyzw",
        "Bc2@yz", "Cd3#za", "De4$ab", "Ef5%bc",
        "Fg6@cd", "Gh7#de", "Hi8$ef", "Ij9%fg",
    ]

    for pw in fallback_candidates:
        if pw not in submitted:
            return pw, "fallback"

    # Random fallback if all else fails
    while True:
        pw = (
            random.choice(string.ascii_uppercase) +
            random.choice(string.ascii_lowercase) +
            str(random.randint(1, 9)) +
            random.choice("@#$%") +
            "".join(random.choices(string.ascii_lowercase, k=random.randint(2, 4)))
        )
        if pw not in submitted:
            return pw, "fallback-random"


# ── Episode runner (your original logic kept) ────────────────────────────────
def run_episode(task: str, person_id: str, llm) -> None:
    rewards: list[float] = []
    steps_taken: int = 0
    success: bool = False
    error_msg: str = "null"
    episode_history: list = []

    print(f"[START] task={task} env=password-policy-env model={MODEL_NAME}", flush=True)

    try:
        PasswordEnvClient = get_env_client()
    except RuntimeError:
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return

    try:
        with PasswordEnvClient(base_url=ENV_BASE_URL) as env:
            obs_obj = env.reset(task=task, person_id=person_id)
            obs = obs_obj.model_dump()

            for step in range(1, MAX_STEPS + 1):
                try:
                    password, action_source = get_agent_action(obs, step, episode_history, llm)
                except Exception as e:
                    password = "Ab1@xy"
                    action_source = "exception-fallback"
                    error_msg = str(e).replace("\n", " ")

                try:
                    obs_obj, reward, done, info = env.step(person_id=person_id, password=password)
                    obs = obs_obj.model_dump()
                    error_msg = "null"
                except Exception as e:
                    reward = 0.0
                    done = True
                    error_msg = str(e).replace("\n", " ")

                was_dup = obs.get("history", [{}])[-1].get("was_duplicate", False) if obs.get("history") else False

                episode_history.append({
                    "step": step,
                    "password": password,
                    "reward": reward,
                    "was_duplicate": was_dup,
                    "source": action_source,
                })

                rewards.append(reward)
                steps_taken = step
                done_str = "true" if done else "false"

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
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps_taken} rewards={rewards_str}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Starting inference | MODEL={MODEL_NAME} | API_BASE_URL={API_BASE_URL[:60] if API_BASE_URL else 'MISSING'}...", flush=True)
    
    llm = make_llm()

    for task in TASKS:
        run_episode(task=task, person_id=f"baseline_agent_{task}", llm=llm)
        print(flush=True)
