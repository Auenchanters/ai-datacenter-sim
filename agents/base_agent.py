import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# Maximum number of retry attempts on a 429 RateLimitError before giving up.
MAX_RATE_LIMIT_RETRIES = 5

# Initial wait in seconds before first retry; doubles each attempt (1, 2, 4, 8, 16s).
RETRY_BASE_DELAY = 1.0


class BaseAgent(ABC):
    """
    Abstract base class for all simulation agents.

    All agents, whether LLM-based or rule-based, must implement
    the decide() method. This method receives the current game state
    as a Python dictionary and must return a valid action dictionary
    containing 'thoughts' and 'actions' keys.

    The base class handles:
        - JSON parsing and validation of agent output.
        - Automatic exponential-backoff retry on RateLimitError (HTTP 429).
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
        Wraps decide() with exponential-backoff retry on RateLimitError,
        then falls back gracefully on any other exception.
        """
        tick = state.get("tick", "?")
        delay = RETRY_BASE_DELAY

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self.decide(state)
                validated = self._validate_response(response)
                self.tick_count += 1
                return validated

            except Exception as e:
                # Check if this is a rate-limit error (HTTP 429).
                is_rate_limit = "RateLimitError" in type(e).__name__ or "429" in str(e)

                if is_rate_limit and attempt < MAX_RATE_LIMIT_RETRIES:
                    wait = delay * (2 ** attempt)
                    print(
                        f"  [RETRY {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}] "
                        f"[{self.model_name}] Tick {tick}: RateLimitError — "
                        f"waiting {wait:.1f}s before retry..."
                    )
                    time.sleep(wait)
                    continue

                # Non-rate-limit error, or retries exhausted — log and give up.
                error_msg = (
                    f"Tick {tick}: Agent '{self.agent_id}' failed - "
                    f"{type(e).__name__}: {str(e)}"
                )
                self.error_log.append(error_msg)
                print(f"  [ERROR] [{self.model_name}] {error_msg}")
                if len(self.error_log) == 1:
                    traceback.print_exc()
                return {"thoughts": f"ERROR: {str(e)}", "actions": []}

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
        }
