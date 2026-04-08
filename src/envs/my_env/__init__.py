"""
client.py — HTTP client for the Password Policy Environment.

This is what users import. It provides a clean, typed Python interface
over the FastAPI server running at ENV_BASE_URL.

Usage:
    from my_env.client import PasswordEnvClient

    client = PasswordEnvClient(base_url="http://localhost:7860")
    obs    = client.reset(task="hard", person_id="agent_001")
    obs, reward, done, info = client.step(
        person_id="agent_001",
        password="_Abc1@"
    )
    state  = client.state()
    client.close()
"""

from __future__ import annotations

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models import Action, Observation, State, Reward


# ─────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:7860"
DEFAULT_TIMEOUT  = 30       # seconds per request
MAX_RETRIES      = 3        # retry on transient network errors


# ─────────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────────

class PasswordEnvClient:
    """
    Typed HTTP client for the Password Policy Environment.

    Wraps all four endpoints:
      reset()  → POST /reset
      step()   → POST /step
      state()  → GET  /state
      health() → GET  /health

    All methods raise requests.HTTPError on non-2xx responses.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

        # Session with automatic retry on transient failures
        self._session = Session()
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://",  adapter)
        self._session.mount("https://", adapter)

    # ── Public API ────────────────────────────────────────────────

    def reset(self, task: str = "hard", person_id: str = "default") -> Observation:
        """
        Start a new episode.

        Args:
            task:      "easy" | "medium" | "hard"
            person_id: Unique identifier for this agent/person.
                       The registry enforces one unique password per person_id.

        Returns:
            Observation — starting state of the episode.
        """
        resp = self._post("/reset", {"task": task, "person_id": person_id})
        return Observation(**resp)

    def step(self, person_id: str, password: str) -> tuple[Observation, float, bool, dict]:
        """
        Submit a candidate password for evaluation.

        Args:
            person_id: Must match the person_id used in reset().
            password:  Candidate password string.

        Returns:
            (observation, reward_value, done, info)

        Note:
            If `password` was already successfully submitted by this person_id
            in a previous episode, a duplicate penalty is applied and
            observation.history[-1].was_duplicate will be True.
        """
        action = Action(person_id=person_id, password=password)
        resp   = self._post("/step", action.model_dump())

        obs    = Observation(**resp["observation"])
        reward = float(resp["reward"])
        done   = bool(resp["done"])
        info   = dict(resp.get("info", {}))

        return obs, reward, done, info

    def state(self) -> State:
        """
        Retrieve full internal environment state (debug/grader use).

        Returns full rule breakdown, hashed password registry stats,
        and episode history. Never call this from the agent loop —
        it exposes information the agent is not supposed to see.
        """
        resp = self._get("/state")
        return State(**resp)

    def health(self) -> bool:
        """
        Liveness check. Returns True if the server is up and healthy.
        """
        try:
            resp = self._get("/health")
            return resp.get("status") == "ok"
        except Exception:
            return False

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()

    # ── Context manager support ───────────────────────────────────

    def __enter__(self) -> "PasswordEnvClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Internal helpers ──────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        url  = f"{self.base_url}{path}"
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        url  = f"{self.base_url}{path}"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
