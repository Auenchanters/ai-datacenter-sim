import json
import os
from typing import Any, Dict, Optional

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from agents.base_agent import BaseAgent
from agents.system_prompt import build_system_prompt


class LLMAgent(BaseAgent):
    """
    A simulation agent powered by any LLM accessible via LiteLLM.

    Supported model strings (pass as model_name):
        - "openai/gpt-4o"
        - "openai/gpt-4o-mini"         (recommended for low-cost testing)
        - "anthropic/claude-3-5-sonnet-20241022"
        - "anthropic/claude-3-haiku-20240307"
        - "gemini/gemini-1.5-pro"
        - "ollama/llama3"              (local, no API key required)
        - "ollama/mistral"

    The agent maintains a rolling message history so the LLM retains
    context of its previous decisions across ticks without exceeding
    the context window limit.
    """

    MAX_HISTORY_TICKS = 5

    def __init__(
        self,
        agent_id: str,
        model_name: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        super().__init__(agent_id=agent_id, model_name=model_name)

        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. Run: pip install litellm"
            )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = build_system_prompt()
        self.message_history: list = []

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the current game state to the LLM and returns the parsed action payload.

        The state is converted to a JSON string and appended to the rolling
        message history as a user message. The LLM response is appended as
        an assistant message to maintain conversational context.
        """
        state_json = json.dumps(state, indent=2)
        user_message = (
            f"TICK {state['tick']} STATE:\n{state_json}\n\n"
            "Analyze the state above and respond with your JSON action payload."
        )

        self.message_history.append({"role": "user", "content": user_message})

        if len(self.message_history) > self.MAX_HISTORY_TICKS * 2:
            self.message_history = self.message_history[-(self.MAX_HISTORY_TICKS * 2):]

        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.message_history

        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content

        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0

        self.message_history.append({"role": "assistant", "content": raw_content})

        return json.loads(raw_content)

    def reset(self):
        self.message_history = []
        self.tick_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.error_log = []
