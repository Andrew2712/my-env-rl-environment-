"""
server/app.py — FastAPI server for the Password Policy Environment.

Endpoints:
  GET  /           → environment info
  GET  /health     → liveness probe (required by HF Spaces)
  POST /reset      → start new episode
  POST /step       → submit password for evaluation
  GET  /state      → full internal state (debug/grader)
  GET  /registry   → password registry stats (admin)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import Observation, Action
from server.environment import PasswordPolicyEnvironment


# ── App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Password Policy Environment",
    description=(
        "OpenEnv-compliant black-box password generation environment. "
        "The agent must satisfy a hidden 5-rule password policy using "
        "only reward signal feedback. Features: per-person unique password "
        "enforcement, SHA-256 hashing, parallel merge sort registry, "
        "duplicate submission penalties."
    ),
    version="1.0.0",
)

# ── Global environment instance ───────────────────────────────────
_env: PasswordPolicyEnvironment | None = None
_current_person_id: str = "default"


# ── Request schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task:      str = "hard"       # "easy" | "medium" | "hard"
    person_id: str = "default"    # unique agent/user identifier


class StepRequest(BaseModel):
    person_id: str                # must match person_id from reset
    password:  str                # candidate password to evaluate


class StepResponse(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        dict
    grade:       float            # current episode grade (0.0–1.0)


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":        "Password Policy Environment",
        "version":     "1.0.0",
        "description": (
            "Black-box password generation environment. "
            "Agent discovers hidden policy via reward signals."
        ),
        "features": [
            "One unique password per person (REQ 1)",
            "Successful password storage + duplicate penalty (REQ 2)",
            "SHA-256 password hashing (REQ 3)",
            "Parallel merge sort for registry search (REQ 4)",
        ],
        "endpoints": {
            "POST /reset":    "Start new episode. Body: {task, person_id}",
            "POST /step":     "Submit password. Body: {person_id, password}",
            "GET  /state":    "Full internal state (debug only)",
            "GET  /registry": "Registry statistics",
            "GET  /health":   "Liveness check",
        },
    }


@app.get("/health")
def health():
    """Liveness probe. HF Spaces requires HTTP 200 here."""
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    The person_id is bound to this episode — all /step calls must
    use the same person_id, and duplicate passwords are tracked
    across episodes for the same person_id.
    """
    global _env, _current_person_id
    try:
        _env = PasswordPolicyEnvironment(task=request.task)
        _current_person_id = request.person_id
        obs  = _env.reset(person_id=request.person_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit a candidate password for evaluation.

    Enforcement:
      - person_id must match the one set in /reset
      - If password was previously successfully submitted by this person,
        a penalty of -0.30 is applied to the reward
      - Successful passwords (raw score >= 0.8) are stored as SHA-256
        hashes in the registry using parallel merge sort

    Returns the observation, reward, done flag, and current grade.
    """
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    try:
        action   = Action(person_id=request.person_id, password=request.password)
        obs, reward, done, info = _env.step(action)
        grade    = _env.grade()
        return StepResponse(
            observation=obs,
            reward=reward.value,
            done=done,
            info=info,
            grade=grade,
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """
    Full internal state including rule breakdown, hashed password
    registry statistics, and complete episode history.
    For debug and grader use only — never call from agent loop.
    """
    global _env
    if _env is None:
        return {"status": "not_initialised", "message": "Call POST /reset first."}
    return _env.state()


@app.get("/registry")
def registry():
    """
    Password registry statistics.
    Shows how many hashed passwords are stored globally and per person.
    Does NOT expose any plaintext or hash values.
    """
    global _env
    if _env is None:
        return {"registry_size": 0, "message": "No active episode."}
    s = _env.state()
    return {
        "registry_size":         s["registry_size"],
        "person_password_count": s["person_password_count"],
        "current_person_id":     s["person_id"],
        "storage_note":          (
            "All passwords stored as SHA-256 hashes only. "
            "Sorted via parallel merge sort for O(log n) binary search."
        ),
    }
