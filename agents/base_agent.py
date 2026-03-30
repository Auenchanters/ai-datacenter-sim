import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# Maximum number of retry attempts on a transient 429 RateLimitError.
# NOTE: Daily quota exhaustion (free-models-per-day) skips retries entirely.
MAX_RATE_LIMIT_RETRIES = 3

# Initial wait in seconds before first retry; doubles each attempt (1, 2, 4s).
RETRY_BASE_DELAY = 1.0

# Strings that indicate a hard daily quota limit has been hit.
# No amount of retrying will fix these — skip immediately.
DAILY_QUOTA_MARKERS = [
    "free-models-per-day",
    "per-day",
    "daily limit",
    "daily quota",
    "tokens-per-day",
    "rate_limit_exceeded",  # Groq daily cap
]


class BaseAgent(ABC):
    """
    Abstract base class for all simulation agents.

    All agents, whether LLM-based or rule-based, must implement
    the decide() method. This method receives the current game state
    as a Python dictionary and must return a valid action dictionary
    containing 'thoughts' and 'actions' keys.

    The base class handles:
        - JSON parsing and validation of agent output.
        - Immediate skip on daily quota exhaustion (Fix 3).
        - Exponential-backoff retry on transient RateLimitError (HTTP 429).
        - Fallback to an empty action list when all retries are exhausted.
        - Session tracking for multi-tick simulations.
    """

    def __init__(self, agent_id: str, model_name: str = "unknown"):
        self.agent_id = agent_id
        self.model_name = model_name
        self.tick_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.error_log: list = []
        self._quota_exhausted: bool = False  # latched True once daily limit hit

    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives the current simulation state dictionary.
        Must return a dictionary with keys:
            'thoughts': str  - the agent's reasoning chain
            'actions':  list - list of action objects to execute
        """
        pass

    def safe_decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wraps decide() with:
          1. Immediate skip if daily quota is already known exhausted.
          2. Immediate skip (no retry) if daily quota error is detected.
          3. Exponential-backoff retry for transient per-minute rate limits.
          4. Graceful fallback on any other exception.
        """
        tick = state.get("tick", "?")

        # Fast path: quota already exhausted earlier in this run.
        if self._quota_exhausted:
            return {"thoughts": "Daily quota exhausted — skipping tick.", "actions": []}

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self.decide(state)
                validated = self._validate_response(response)
                self.tick_count += 1
                return validated

            except Exception as e:
                err_str = str(e)

                # Fix 3: Detect hard daily quota exhaustion — skip immediately, no retry.
                is_daily_quota = any(marker in err_str.lower() for marker in DAILY_QUOTA_MARKERS)
                if is_daily_quota:
                    self._quota_exhausted = True
                    msg = f"Tick {tick}: Daily API quota exhausted. All remaining ticks will be skipped."
                    self.error_log.append(msg)
                    print(f"  [QUOTA EXHAUSTED] [{self.model_name}] {msg}")
                    print(f"  [QUOTA EXHAUSTED] Reset time is in the error above. Re-run tomorrow or add credits.")
                    return {"thoughts": "Daily quota exhausted.", "actions": []}

                # Transient rate limit (per-minute / per-second throttle) — retry with backoff.
                is_rate_limit = "RateLimitError" in type(e).__name__ or "429" in err_str

                if is_rate_limit and attempt < MAX_RATE_LIMIT_RETRIES:
                    wait = RETRY_BASE_DELAY * (2 ** attempt)
                    print(
                        f"  [RETRY {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}] "
                        f"[{self.model_name}] Tick {tick}: Transient rate limit — "
                        f"waiting {wait:.1f}s before retry..."
                    )
                    time.sleep(wait)
                    continue

                # Non-rate-limit error, or transient retries exhausted.
                error_msg = (
                    f"Tick {tick}: Agent '{self.agent_id}' failed - "
                    f"{type(e).__name__}: {err_str}"
                )
                self.error_log.append(error_msg)
                print(f"  [ERROR] [{self.model_name}] {error_msg}")
                if len(self.error_log) == 1:
                    traceback.print_exc()
                return {"thoughts": f"ERROR: {err_str}", "actions": []}

    def _validate_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, str):
            response = json.loads(response)

        if not isinstance(response, dict):
            raise ValueError(f"Agent response must be a JSON object. Got: {type(response)}.")

        if "actions" not in response:
            response["actions"] = []

        if "thoughts" not in response:
            response["thoughts"] = ""

        if not isinstance(response["actions"], list):
            raise ValueError("'actions' field must be a JSON array.")

        return response

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "ticks_completed": self.tick_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "errors": len(self.error_log),
            "quota_exhausted": self._quota_exhausted,
        }
