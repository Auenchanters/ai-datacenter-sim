import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all simulation agents.
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
        pass

    def safe_decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wraps decide() with error handling.
        Prints the full exception immediately so it appears in the terminal.
        """
        try:
            response = self.decide(state)
            validated = self._validate_response(response)
            self.tick_count += 1
            return validated
        except Exception as e:
            import traceback
            tick = state.get('tick', '?')
            error_msg = f"Tick {tick}: Agent '{self.agent_id}' failed - {type(e).__name__}: {str(e)}"
            self.error_log.append(error_msg)
            # Print immediately so the error is visible in the terminal
            print(f"  [ERROR] [{self.model_name}] {error_msg}")
            if len(self.error_log) == 1:
                # Print full traceback on the very first error only
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
