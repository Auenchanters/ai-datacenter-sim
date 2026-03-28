import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all simulation agents.

    All agents, whether LLM-based or rule-based, must implement
    the decide() method. This method receives the current game state
    as a Python dictionary and must return a valid action dictionary
    containing 'thoughts' and 'actions' keys.

    The base class handles:
        - JSON parsing and validation of agent output.
        - Fallback to an empty action list on malformed responses.
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
            'actions':  list - list of command dictionaries
        """
        pass

    def safe_decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wraps decide() with error handling.
        If the agent returns malformed output or raises an exception,
        a safe empty action response is returned and the error is logged.
        """
        try:
            response = self.decide(state)
            validated = self._validate_response(response)
            self.tick_count += 1
            return validated
        except Exception as e:
            error_msg = f"Tick {state.get('tick', '?')}: Agent '{self.agent_id}' failed - {str(e)}"
            self.error_log.append(error_msg)
            return {"thoughts": f"ERROR: {str(e)}", "actions": []}

    def _validate_response(self, response: Any) -> Dict[str, Any]:
        """
        Ensures the agent response is a dictionary with the required keys.
        If the response is a raw JSON string, attempts to parse it first.
        """
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
