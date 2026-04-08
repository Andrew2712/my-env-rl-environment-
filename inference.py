"""
inference.py — Baseline agent for the Password Policy Environment.

Uses client.py (PasswordEnvClient) to interact with the environment.
Uses the OpenAI SDK to drive the LLM agent.

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
    """Load a .py file as a named module by absolute path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # register BEFORE exec so cross-imports resolve
    spec.loader.exec_module(mod)
    return mod

# models.py must be loaded BEFORE client.py (client imports from models)
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
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = 10

# ── OpenAI client ──────────────────────────────────────────────────────────────

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert password generation agent in a black-box \
optimisation environment.

TASK:
Generate a password satisfying a completely hidden policy. After each attempt \
you receive a reward score between 0.0 and 1.0. A score of 1.0 means all rules \
are satisfied. You are NEVER told what the rules are.

IMPORTANT RULES YOU MUST FOLLOW:
- Never repeat a password you have already submitted. Repeated passwords incur \
a -0.30 penalty.
- Each password must be a fresh, unique attempt.

STRATEGY:
1. First attempt: use a diverse password covering length ~6, mixed case, digit, \
symbol (@#$%), starting with a letter or underscore.
2. After each reward, reason about which aspect caused the score to change.
3. Make one targeted change per attempt to isolate rule effects.
4. Track what has and has not worked.
5. Never repeat a previously submitted password.

COMMON POLICY PATTERNS TO TEST:
  - Length (5-7 chars)
  - Uppercase + lowercase mix
  - At least one digit, not all digits
  - At least one symbol from @#$%, not all symbols
  - First character: letter or underscore only

OUTPUT FORMAT:
Respond ONLY with a JSON object - no markdown, no explanation:
{"password": "YourPasswordHere", "reasoning": "one line rationale"}
"""


def build_prompt(obs: dict, step: int) -> str:
    history = obs.get("history", [])
    lines   = []
    for r in history:
        dup = " [DUPLICATE - penalised]" if r.get("was_duplicate") else ""
        lines.append(
            f"  step={r['step']} password='{r['password']}' "
            f"reward={r['reward']:.2f}{dup}"
        )
    history_str = "\n".join(lines) if lines else "  (no attempts yet)"

    return (
        f"Step {step} of {MAX_STEPS}\n"
        f"Last password : '{obs.get('last_password', '')}'\n"
        f"Last reward   : {obs.get('last_reward', 0.0):.2f}\n"
        f"Best so far   : {obs.get('best_reward_so_far', 0.0):.2f}\n"
        f"Steps left    : {obs.get('steps_remaining', 0)}\n\n"
        f"Attempt history:\n{history_str}\n\n"
        "What is your next password? Respond ONLY with the JSON object."
    )


def get_agent_action(obs: dict, step: int) -> str:
    """Call LLM, parse response, return password string."""
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs, step)},
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
        if pw:
            return pw
    except Exception:
        pass

    fallbacks = [
        "_Abc1@", "_Dev2#", "_Run3$", "_Fix4%", "_Try5@",
        "_Go6#",  "_Win7$", "_End8%", "_Set9@", "_Top1#",
    ]
    return fallbacks[min(step - 1, len(fallbacks) - 1)]


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task: str, person_id: str) -> None:
    rewards:     list[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    error_msg:   str         = "null"

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
                    password  = get_agent_action(obs, step)
                    error_msg = "null"
                except Exception as e:
                    password  = "_Fallback1@"
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
        run_episode(task=task, person_id=f"baseline_agent_{task}")
        print(flush=True)