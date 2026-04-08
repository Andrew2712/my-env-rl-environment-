"""
models.py — Shared type definitions for the Password Policy Environment.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class Action(BaseModel):
    person_id: str = Field(..., min_length=1)
    password: str  = Field(...)


class AttemptRecord(BaseModel):
    step: int
    password: str
    reward: float
    was_duplicate: bool = False


class Observation(BaseModel):
    task: str
    person_id: str
    attempt_number: int
    steps_remaining: int
    last_password: str
    last_reward: float
    best_reward_so_far: float
    history: list[AttemptRecord]
    message: str


class RuleScore(BaseModel):
    rule_id: str
    description: str
    score: float
    weight: float


class State(BaseModel):
    task: str
    person_id: str
    step_count: int
    max_steps: int
    done: bool
    last_password: str
    last_reward: float
    best_reward_so_far: float
    rule_breakdown: list[RuleScore]
    history: list[AttemptRecord]
    registry_size: int
    person_password_count: int
    policy_hint: str


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0)
    step: int
    rule_scores: dict[str, float]
    is_duplicate: bool = False
    penalty_applied: float = 0.0