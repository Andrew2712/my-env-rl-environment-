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


app = FastAPI(
    title="Password Policy Environment",
    version="1.0.0",
)

_env: PasswordPolicyEnvironment | None = None
_current_person_id: str = "default"

# Store last grade per task so validator can retrieve it after episode ends
_last_grades: dict[str, float] = {
    "easy":   0.001,
    "medium": 0.001,
    "hard":   0.001,
}


def clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1 — never exactly 0.0 or 1.0."""
    return round(max(0.001, min(0.999, float(score))), 4)


class ResetRequest(BaseModel):
    task:      str = "hard"
    person_id: str = "default"


class StepRequest(BaseModel):
    person_id: str
    password:  str


class StepResponse(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        dict
    grade:       float


@app.get("/")
def root():
    return {
        "name":        "Password Policy Environment",
        "version":     "1.0.0",
        "description": "Black-box password generation environment. Agent discovers hidden policy via reward signals.",
        "features": [
            "One unique password per person (REQ 1)",
            "Successful password storage + duplicate penalty (REQ 2)",
            "SHA-256 password hashing (REQ 3)",
            "Parallel merge sort for registry search (REQ 4)",
        ],
        "endpoints": {
            "POST /reset":          "Start new episode. Body: {task, person_id}",
            "POST /step":           "Submit password. Body: {person_id, password}",
            "GET  /state":          "Full internal state (debug only)",
            "GET  /registry":       "Registry statistics",
            "GET  /health":         "Liveness check",
            "GET  /grade":          "Current episode grade (0.001-0.999)",
            "GET  /grade/{task_id}":"Grade for specific task (0.001-0.999)",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()):
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
    global _env, _last_grades
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )
    try:
        action = Action(person_id=request.person_id, password=request.password)
        obs, reward, done, info = _env.step(action)

        reward_value = clamp_score(reward.value)
        grade_value  = clamp_score(_env.grade())

        # Store last grade for this task so validator can retrieve it
        _last_grades[_env._task] = grade_value

        return StepResponse(
            observation=obs,
            reward=reward_value,
            done=done,
            info=info,
            grade=grade_value,
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    global _env
    if _env is None:
        return {"status": "not_initialised", "message": "Call POST /reset first."}
    s = _env.state()
    # Clamp all score fields in state
    if "last_reward" in s:
        s["last_reward"] = clamp_score(s["last_reward"]) if s["last_reward"] else 0.001
    if "best_reward_so_far" in s:
        s["best_reward_so_far"] = clamp_score(s["best_reward_so_far"]) if s["best_reward_so_far"] else 0.001
    return s


@app.get("/grade")
def grade():
    """Return current episode grade strictly in (0.001, 0.999)."""
    global _env, _last_grades
    if _env is None:
        return {"grade": 0.001}
    grade_value = clamp_score(_env.grade())
    _last_grades[_env._task] = grade_value
    return {"grade": grade_value}


@app.get("/grade/{task_id}")
def grade_by_task(task_id: str):
    """Return grade for a specific task strictly in (0.001, 0.999)."""
    global _env, _last_grades
    # If current env matches requested task, compute fresh grade
    if _env is not None and _env._task == task_id:
        grade_value = clamp_score(_env.grade())
        _last_grades[task_id] = grade_value
        return {"task": task_id, "grade": grade_value}
    # Otherwise return last known grade for this task
    grade_value = _last_grades.get(task_id, 0.001)
    return {"task": task_id, "grade": grade_value}


@app.get("/registry")
def registry():
    global _env
    if _env is None:
        return {"registry_size": 0, "message": "No active episode."}
    s = _env.state()
    return {
        "registry_size":         s["registry_size"],
        "person_password_count": s["person_password_count"],
        "current_person_id":     s["person_id"],
        "registry_storage_path": s.get("registry_storage_path", "unknown"),
        "storage_note": (
            "All passwords stored as SHA-256 hashes only. "
            "Persisted to registry.json on every successful registration. "
            "Sorted via parallel merge sort for O(log n) binary search."
        ),
    }