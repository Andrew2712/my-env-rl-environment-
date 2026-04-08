"""
server/environment.py — Password Policy Environment (core logic)

Implements all four new requirements:

  REQ 1 — One unique password per person:
           PasswordRegistry enforces this per person_id.

  REQ 2 — Store successful passwords + duplicate penalty:
           Registry stores hashed successful passwords.
           Duplicate submission → reward penalty of -0.30.

  REQ 3 — Password hashing:
           hashlib.sha256 used before storage.
           Plaintext never persists in the registry.

  REQ 4 — Parallel merge sort for search:
           parallel_merge_sort() uses ThreadPoolExecutor to sort
           chunks concurrently. binary_search() finds duplicates
           in O(log n) on the sorted hash list.
"""

from __future__ import annotations

import sys
import os
import json
import hashlib
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Allow import of models.py from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    Action, Observation, State, Reward,
    AttemptRecord, RuleScore,
)


# ═════════════════════════════════════════════════════════════════
# REQUIREMENT 4 — PARALLEL MERGE SORT + BINARY SEARCH
# Used by PasswordRegistry to maintain a sorted list of hashed
# passwords and search it efficiently.
# ═════════════════════════════════════════════════════════════════

def _merge(left: list[str], right: list[str]) -> list[str]:
    """
    Standard two-pointer merge of two sorted lists.
    Operates on SHA-256 hex strings — lexicographic order is correct
    for hash comparison because hex chars are uniform in distribution.
    """
    result: list[str] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def _merge_sort_sequential(arr: list[str]) -> list[str]:
    """
    Standard recursive merge sort — used for small chunks inside
    parallel workers (avoids spawning threads within threads).
    """
    if len(arr) <= 1:
        return arr
    mid   = len(arr) // 2
    left  = _merge_sort_sequential(arr[:mid])
    right = _merge_sort_sequential(arr[mid:])
    return _merge(left, right)


def parallel_merge_sort(arr: list[str], max_workers: int = 4) -> list[str]:
    """
    Parallel merge sort using ThreadPoolExecutor.

    Strategy:
      1. Split the input list into `max_workers` roughly equal chunks.
      2. Sort each chunk concurrently in a thread pool.
      3. Sequentially merge all sorted chunks back into one sorted list.

    Why threads (not processes): GIL is acceptable here because
    the workload is comparison-heavy (pure Python string ops), not
    CPU-bound computation. For very large lists, switch to
    multiprocessing.Pool with picklable data.

    Time complexity: O((n/k) log(n/k)) per thread × O(n log k) merge
    where k = max_workers.
    """
    if len(arr) <= 1:
        return list(arr)

    # Split into chunks — cap at actual list length
    k          = min(max_workers, len(arr))
    chunk_size = max(1, len(arr) // k)
    chunks     = [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

    # Sort each chunk in a separate thread
    sorted_chunks: list[list[str]] = [[] for _ in chunks]
    with ThreadPoolExecutor(max_workers=k) as executor:
        future_to_idx = {
            executor.submit(_merge_sort_sequential, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            sorted_chunks[idx] = future.result()

    # Sequential k-way merge
    result = sorted_chunks[0]
    for chunk in sorted_chunks[1:]:
        result = _merge(result, chunk)

    return result


def binary_search(sorted_arr: list[str], target: str) -> bool:
    """
    Binary search on a sorted list of SHA-256 hex strings.
    Returns True if target is found, False otherwise.
    O(log n) — efficient for large registries.
    """
    lo, hi = 0, len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == target:
            return True
        elif sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return False


# ═════════════════════════════════════════════════════════════════
# REQUIREMENT 3 — PASSWORD HASHING
# ═════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256.

    SHA-256 is used here because:
      - It is deterministic (same input → same hash always, required
        for duplicate detection)
      - It is available in Python stdlib (hashlib) — no dependencies
      - It produces a fixed 64-char hex string suitable for sorting

    For production auth systems, use bcrypt or Argon2 with a salt.
    This environment uses SHA-256 because the goal is duplicate
    detection across submissions, not authentication security.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ═════════════════════════════════════════════════════════════════
# REQUIREMENTS 1 & 2 — PASSWORD REGISTRY
# One unique password per person. Stores hashed successes.
# Uses parallel merge sort to maintain sorted storage.
# ═════════════════════════════════════════════════════════════════

class PasswordRegistry:
    """
    Thread-safe, file-persisted registry.

    Maps person_id → sorted list of SHA-256 hex hashes.

    Storage layout (JSON file):
    {
        "agent_001": ["0a1b2c...", "3d4e5f...", ...],
        "agent_002": ["7f8a9b...", ...]
    }

    All keys are SHA-256 hashes — no plaintext is ever written.
    The file is read on __init__ and written atomically on every
    register() call using a temp-file + os.replace() pattern,
    so a crash mid-write never corrupts existing data.

    Operations:
      is_duplicate(person_id, password) → bool   [O(log n), binary search]
      register(person_id, password)     → None    [O(n log n), parallel sort + file write]
      person_count(person_id)           → int
      total_stored()                    → int
    """

    # Default path: written next to this file inside server/
    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "registry.json")

    def __init__(self, storage_path: str | None = None) -> None:
        self._path  = storage_path or self.DEFAULT_PATH
        self._lock  = threading.Lock()
        self._store: dict[str, list[str]] = self._load()

    # ── Persistence helpers ───────────────────────────────────────

    def _load(self) -> dict[str, list[str]]:
        """
        Load registry from JSON file at startup.
        Returns an empty dict if the file does not exist yet
        (first run) or is malformed.
        """
        if not os.path.exists(self._path):
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validate structure: must be dict[str, list[str]]
            if isinstance(data, dict):
                return {
                    k: v for k, v in data.items()
                    if isinstance(k, str) and isinstance(v, list)
                }
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save(self) -> None:
        """
        Write the current in-memory store to disk atomically.

        Uses temp-file + os.replace():
          1. Write to registry.json.tmp
          2. os.replace() swaps it in atomically
        This guarantees the file is never left in a half-written state.

        Called inside the lock — caller must hold self._lock.
        """
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)
            os.replace(tmp_path, self._path)     # atomic on POSIX and Windows
        except OSError as e:
            # Non-fatal: in-memory state remains correct.
            # Log the error but do not crash the environment.
            print(
                f"[PasswordRegistry] WARNING: could not save registry to "
                f"{self._path}: {e}",
                flush=True,
            )

    # ── Public API ────────────────────────────────────────────────

    def is_duplicate(self, person_id: str, password: str) -> bool:
        """
        Check if this person has already successfully submitted this password.

        1. Hash plaintext → SHA-256 hex
        2. Retrieve person's sorted hash list (from memory, not disk)
        3. Binary search — O(log n)
        """
        pw_hash = hash_password(password)
        with self._lock:
            history = list(self._store.get(person_id, []))
        return binary_search(history, pw_hash)

    def register(self, person_id: str, password: str) -> None:
        """
        Store a successful password hash for a person.

        1. Hash plaintext → SHA-256 hex
        2. Append to person's in-memory list
        3. Re-sort using parallel_merge_sort (keeps binary search valid)
        4. Write entire registry to registry.json atomically

        Idempotent — re-registering the same password is safe.
        is_duplicate() is called upstream to prevent it, but _save()
        is stable regardless.
        """
        pw_hash = hash_password(password)
        with self._lock:
            current = self._store.get(person_id, [])
            current.append(pw_hash)
            self._store[person_id] = parallel_merge_sort(current)
            self._save()    # ← persist to disk inside the lock

    def person_count(self, person_id: str) -> int:
        """Number of unique hashed passwords stored for this person."""
        with self._lock:
            return len(self._store.get(person_id, []))

    def total_stored(self) -> int:
        """Total hashed passwords across all persons."""
        with self._lock:
            return sum(len(v) for v in self._store.values())

    @property
    def storage_path(self) -> str:
        """Absolute path of the JSON registry file."""
        return os.path.abspath(self._path)


# Singleton registry — shared across all episodes on this server process.
# Loads any previously stored hashes from registry.json on startup.
_REGISTRY = PasswordRegistry()


# ═════════════════════════════════════════════════════════════════
# HIDDEN POLICY CONSTANTS
# ═════════════════════════════════════════════════════════════════

MIN_LEN        = 5
MAX_LEN        = 7
VALID_SYMBOLS  = set("@#$%")
ALLOWED_FIRST  = set(string.ascii_letters + "_")
RULE_WEIGHT    = 0.20       # each of the 5 rules contributes equally

DUPLICATE_PENALTY = 0.30    # deducted from raw reward on duplicate submission


# ═════════════════════════════════════════════════════════════════
# POLICY RULE SCORERS (hidden from agent)
# ═════════════════════════════════════════════════════════════════

def _score_length(pw: str) -> float:
    """Rule 1: Length must be 5–7 characters."""
    n = len(pw)
    if n == 0:
        return 0.0
    if MIN_LEN <= n <= MAX_LEN:
        return 1.0
    if n < MIN_LEN:
        return n / MIN_LEN
    return max(0.0, 1.0 - (n - MAX_LEN) * 0.20)


def _score_case_mix(pw: str) -> float:
    """Rule 2: Must have at least one uppercase and one lowercase letter."""
    has_upper = any(c.isupper() for c in pw)
    has_lower = any(c.islower() for c in pw)
    return (0.5 if has_upper else 0.0) + (0.5 if has_lower else 0.0)


def _score_digits(pw: str) -> float:
    """Rule 3: At least one digit, but not all digits."""
    if not pw:
        return 0.0
    n_digits   = sum(c.isdigit() for c in pw)
    all_digits = (n_digits == len(pw))
    if all_digits:
        return 0.0
    return 1.0 if n_digits >= 1 else 0.0


def _score_symbols(pw: str) -> float:
    """Rule 4: At least one symbol from {@ # $ %}, but not all symbols."""
    if not pw:
        return 0.0
    n_sym  = sum(c in VALID_SYMBOLS for c in pw)
    all_sym = all(c in VALID_SYMBOLS for c in pw)
    if all_sym:
        return 0.0
    return 1.0 if n_sym >= 1 else 0.0


def _score_first_char(pw: str) -> float:
    """Rule 5: First character must be a letter or underscore."""
    if not pw:
        return 0.0
    return 1.0 if pw[0] in ALLOWED_FIRST else 0.0


# ═════════════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ═════════════════════════════════════════════════════════════════

def compute_reward(
    password: str,
    step: int,
    person_id: str,
) -> Reward:
    """
    Compute reward for a submitted password.

    Pipeline:
      1. Check registry for duplicate (REQ 1 & 2)
      2. Compute per-rule scores
      3. Apply duplicate penalty if needed (REQ 2)
      4. Return Reward with full breakdown (internal) + final value
    """
    # ── Duplicate check ──────────────────────────────────────────
    is_dup = _REGISTRY.is_duplicate(person_id, password)

    # ── Rule scores ──────────────────────────────────────────────
    r1 = _score_length(password)
    r2 = _score_case_mix(password)
    r3 = _score_digits(password)
    r4 = _score_symbols(password)
    r5 = _score_first_char(password)

    rule_scores = {
        "rule_1_length":     r1,
        "rule_2_case_mix":   r2,
        "rule_3_digits":     r3,
        "rule_4_symbols":    r4,
        "rule_5_first_char": r5,
    }

    raw_total = RULE_WEIGHT * (r1 + r2 + r3 + r4 + r5)

    # ── Duplicate penalty (REQ 2) ─────────────────────────────────
    penalty = DUPLICATE_PENALTY if is_dup else 0.0
    final   = round(max(0.0, min(1.0, raw_total - penalty)), 4)

    return Reward(
        value=final,
        step=step,
        rule_scores=rule_scores,
        is_duplicate=is_dup,
        penalty_applied=penalty,
    )


# ═════════════════════════════════════════════════════════════════
# TASK GRADERS
# Deterministic, no LLM calls, return float in [0.0, 1.0].
# ═════════════════════════════════════════════════════════════════

def grade_easy(history: list[AttemptRecord]) -> float:
    """Easy: score of the first submission."""
    if not history:
        return 0.0
    return history[0].reward


def grade_medium(history: list[AttemptRecord]) -> float:
    """
    Medium: weighted combination of
      best reward achieved (60%) +
      improvement from first to best (20%) +
      step efficiency (20%).
    Duplicate attempts reduce score because they waste steps.
    """
    if not history:
        return 0.0
    best_reward  = max(r.reward for r in history)
    first_reward = history[0].reward
    improvement  = max(0.0, best_reward - first_reward)
    best_step    = next(i + 1 for i, r in enumerate(history) if r.reward == best_reward)
    efficiency   = 1.0 - ((best_step - 1) / 10)
    score = (0.60 * best_reward) + (0.20 * improvement) + (0.20 * efficiency)
    return round(min(1.0, max(0.0, score)), 4)


def grade_hard(history: list[AttemptRecord]) -> float:
    """
    Hard: bonus structure for reaching perfect score.
      Perfect achieved → 0.50 + 0.30 + 0.20 × efficiency
      Otherwise        → 0.50 × best_reward + 0.20 × efficiency
    Duplicate penalties visible in history reduce best_reward,
    making this task harder for agents that repeat submissions.
    """
    if not history:
        return 0.0
    best_reward    = max(r.reward for r in history)
    reached_perfect = any(r.reward >= 1.0 for r in history)
    best_step      = next(i + 1 for i, r in enumerate(history) if r.reward == best_reward)
    efficiency     = 1.0 - ((best_step - 1) / 10)
    if reached_perfect:
        score = 0.50 + 0.30 + (0.20 * efficiency)
    else:
        score = (0.50 * best_reward) + (0.20 * efficiency)
    return round(min(1.0, max(0.0, score)), 4)


# ═════════════════════════════════════════════════════════════════
# TASK CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "easy": {
        "name":              "validate_password",
        "description":       (
            "A starting password is provided. Evaluate it and try to "
            "improve it against the hidden policy."
        ),
        "starting_password": "hello",
        "max_steps":         10,
        "grader":            grade_easy,
    },
    "medium": {
        "name":              "fix_password",
        "description":       (
            "A non-compliant password is given. Fix it iteratively "
            "using only the reward signal as guidance."
        ),
        "starting_password": "HELLO123",
        "max_steps":         10,
        "grader":            grade_medium,
    },
    "hard": {
        "name":              "generate_compliant_password",
        "description":       (
            "No starting password. Generate a fully policy-compliant "
            "password from scratch using reward signal feedback only. "
            "Repeated passwords incur a penalty."
        ),
        "starting_password": "",
        "max_steps":         10,
        "grader":            grade_hard,
    },
}


# ═════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS
# ═════════════════════════════════════════════════════════════════

class PasswordPolicyEnvironment:
    """
    OpenEnv-compliant environment with:
      - Hidden policy (agent infers from rewards)
      - Per-person unique password enforcement (REQ 1)
      - Hashed password storage (REQ 3)
      - Parallel merge sort for registry search (REQ 4)
      - Duplicate submission penalty (REQ 2)
    """

    def __init__(self, task: str = "hard") -> None:
        if task not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task}'. Valid: {list(TASK_CONFIGS)}"
            )
        self._task      = task
        self._config    = TASK_CONFIGS[task]
        self._person_id = "default"
        self._history:  list[AttemptRecord] = []
        self._step_count  = 0
        self._max_steps   = self._config["max_steps"]
        self._done        = False
        self._last_reward: Reward | None = None
        self._last_password = self._config["starting_password"]

    # ── OpenEnv required methods ──────────────────────────────────

    def reset(self, person_id: str = "default") -> Observation:
        """Initialise a fresh episode for the given person."""
        self._person_id     = person_id
        self._history       = []
        self._step_count    = 0
        self._done          = False
        self._last_password = self._config["starting_password"]
        self._last_reward   = None

        return Observation(
            task=self._task,
            person_id=self._person_id,
            attempt_number=0,
            steps_remaining=self._max_steps,
            last_password=self._last_password,
            last_reward=0.0,
            best_reward_so_far=0.0,
            history=[],
            message=(
                f"Episode started | task={self._config['name']} | "
                f"person_id={person_id} | "
                f"{self._config['description']} | "
                f"{self._max_steps} attempts available."
            ),
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Evaluate submitted password.
        Enforces unique-password rule and applies penalties.
        Registers successful passwords in hashed form.
        """
        if self._done:
            raise RuntimeError(
                "Episode finished. Call reset() to start a new episode."
            )
        if action.person_id != self._person_id:
            raise ValueError(
                f"person_id mismatch: expected '{self._person_id}', "
                f"got '{action.person_id}'."
            )

        self._step_count += 1
        password = action.password

        # ── Compute reward (includes duplicate check) ─────────────
        reward = compute_reward(password, self._step_count, self._person_id)

        # ── Register successful passwords (REQ 2) ─────────────────
        # A password is "successful" if it scores >= 0.8 raw AND is not a dup.
        # We store before penalty so the registry reflects genuine achievements.
        raw_score = RULE_WEIGHT * sum(reward.rule_scores.values())
        if raw_score >= 0.80 and not reward.is_duplicate:
            _REGISTRY.register(self._person_id, password)

        # ── Record attempt ────────────────────────────────────────
        record = AttemptRecord(
            step=self._step_count,
            password=password,
            reward=reward.value,
            was_duplicate=reward.is_duplicate,
        )
        self._history.append(record)
        self._last_password = password
        self._last_reward   = reward

        # ── Done conditions ───────────────────────────────────────
        done = (reward.value >= 1.0) or (self._step_count >= self._max_steps)
        self._done = done

        best_so_far = max(r.reward for r in self._history)

        # ── Message ───────────────────────────────────────────────
        if reward.is_duplicate:
            msg = (
                f"⚠ DUPLICATE PASSWORD — penalty -{DUPLICATE_PENALTY:.2f} applied. "
                f"Reward after penalty: {reward.value:.2f}. "
                "Each person may only use each password once."
            )
        elif reward.value >= 1.0:
            msg = "✓ Perfect score! All policy rules satisfied. Episode complete."
        elif done:
            msg = f"Step budget exhausted. Best reward: {best_so_far:.2f}."
        else:
            left = self._max_steps - self._step_count
            msg = (
                f"Attempt {self._step_count}: reward={reward.value:.2f}. "
                f"{left} attempts remaining."
            )

        obs = Observation(
            task=self._task,
            person_id=self._person_id,
            attempt_number=self._step_count,
            steps_remaining=self._max_steps - self._step_count,
            last_password=password,
            last_reward=reward.value,
            best_reward_so_far=best_so_far,
            history=self._history,
            message=msg,
        )

        info = {
            "task_name":         self._config["name"],
            "step":              self._step_count,
            "done":              done,
            "was_duplicate":     reward.is_duplicate,
            "penalty_applied":   reward.penalty_applied,
            "raw_score_before_penalty": round(raw_score, 4),
            "registry_size":     _REGISTRY.total_stored(),
        }

        return obs, reward, done, info

    def state(self) -> dict:
        """Full internal state including rule breakdown and registry stats."""
        rule_breakdown = []
        descriptions   = {
            "rule_1_length":     "Length 5–7 characters",
            "rule_2_case_mix":   "Both uppercase and lowercase",
            "rule_3_digits":     "At least one digit, not all digits",
            "rule_4_symbols":    "At least one symbol (@#$%), not all symbols",
            "rule_5_first_char": "First char is letter or underscore",
        }
        if self._last_reward:
            for rule_id, score in self._last_reward.rule_scores.items():
                rule_breakdown.append(RuleScore(
                    rule_id=rule_id,
                    description=descriptions.get(rule_id, ""),
                    score=score,
                    weight=RULE_WEIGHT,
                ).model_dump())

        return {
            "task":                  self._task,
            "task_name":             self._config["name"],
            "person_id":             self._person_id,
            "step_count":            self._step_count,
            "max_steps":             self._max_steps,
            "done":                  self._done,
            "last_password":         self._last_password,
            "last_password_hash":    hash_password(self._last_password) if self._last_password else "",
            "last_reward":           self._last_reward.value if self._last_reward else 0.0,
            "was_last_duplicate":    self._last_reward.is_duplicate if self._last_reward else False,
            "best_reward_so_far":    max((r.reward for r in self._history), default=0.0),
            "rule_breakdown":        rule_breakdown,
            "history":               [r.model_dump() for r in self._history],
            "registry_size":         _REGISTRY.total_stored(),
            "person_password_count": _REGISTRY.person_count(self._person_id),
            "registry_storage_path": _REGISTRY.storage_path,
            "policy_hint":           "Hidden — agent must infer from reward signal only.",
        }

    def grade(self) -> float:
        """Run the task grader on current episode history."""
        return self._config["grader"](self._history)

    def close(self) -> None:
        """No-op — satisfies OpenEnv spec close() requirement."""
        pass
