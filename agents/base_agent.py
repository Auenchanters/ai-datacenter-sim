import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

MAX_RATE_LIMIT_RETRIES = 3
RETRY_BASE_DELAY = 1.0

# Strings that mean the daily/total quota is PERMANENTLY gone — no retry will help.
# NOTE: "rate_limit_exceeded" is intentionally excluded here — that is a
# per-minute throttle (transient) and should be retried, not fast-failed.
# Groq returns RateLimitError / 429 for per-minute limits, which is handled
# separately by the is_rate_limit branch below.
DAILY_QUOTA_MARKERS = [
    "free-models-per-day",
    "per-day",
    "daily limit",
    "daily quota",
    "tokens-per-day",
    "resource_exhausted",   # Gemini RESOURCE_EXHAUSTED daily cap
    "limit: 0",             # Gemini reports 'limit: 0' when daily quota = 0
    "per day per project",  # Gemini GenerateRequestsPerDayPerProject
    "GenerateRequestsPerDay",
]


class BaseAgent(ABC):
    """
    Abstract base class for all simulation agents.
    Handles quota fast-fail and transient rate-limit retry.
    """

    def __init__(self, agent_id: str, model_name: str = "unknown"):
        self.agent_id = agent_id
        self.model_name = model_name
        self.tick_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.error_log: list = []
        self._quota_exhausted: bool = False

    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def safe_decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        tick = state.get("tick", "?")

        if self._quota_exhausted:
            return {"thoughts": "Daily quota exhausted — skipping tick.", "actions": []}

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response  = self.decide(state)
                validated = self._validate_response(response)
                self.tick_count += 1
                return validated

            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()

                # Hard daily quota — fast-fail immediately, no retry.
                is_daily = any(m.lower() in err_lower for m in DAILY_QUOTA_MARKERS)
                if is_daily:
                    self._quota_exhausted = True
                    msg = f"Tick {tick}: Daily quota exhausted. Skipping all remaining ticks."
                    self.error_log.append(msg)
                    print(f"  [QUOTA EXHAUSTED] [{self.model_name}] {msg}")
                    return {"thoughts": "Daily quota exhausted.", "actions": []}

                # Transient per-minute throttle — retry with backoff.
                is_rate_limit = (
                    "RateLimitError" in type(e).__name__
                    or "429" in err_str
                    or "rate_limit_exceeded" in err_lower
                )
                if is_rate_limit and attempt < MAX_RATE_LIMIT_RETRIES:
                    wait = RETRY_BASE_DELAY * (2 ** attempt)
                    print(
                        f"  [RETRY {attempt+1}/{MAX_RATE_LIMIT_RETRIES}] "
                        f"[{self.model_name}] Tick {tick}: throttle — waiting {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue

                error_msg = f"Tick {tick}: '{self.agent_id}' failed - {type(e).__name__}: {err_str}"
                self.error_log.append(error_msg)
                print(f"  [ERROR] [{self.model_name}] {error_msg}")
                if len(self.error_log) == 1:
                    traceback.print_exc()
                return {"thoughts": f"ERROR: {err_str}", "actions": []}

    def _validate_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, dict):
            raise ValueError(f"Response must be a JSON object. Got: {type(response)}.")
        response.setdefault("actions", [])
        response.setdefault("thoughts", "")
        if not isinstance(response["actions"], list):
            raise ValueError("'actions' must be a JSON array.")
        return response

    def get_stats(self) -> Dict[str, Any]:
        return {
            "agent_id":                self.agent_id,
            "model_name":              self.model_name,
            "ticks_completed":         self.tick_count,
            "total_prompt_tokens":     self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "errors":                  len(self.error_log),
            "quota_exhausted":         self._quota_exhausted,
        }
