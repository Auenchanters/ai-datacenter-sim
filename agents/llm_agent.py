import json
import os
from typing import Any, Dict, Optional

try:
    import litellm
    litellm.suppress_debug_info = True
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from agents.base_agent import BaseAgent
from agents.system_prompt import build_system_prompt


class LLMAgent(BaseAgent):
    """
    A simulation agent powered by any LLM accessible via LiteLLM.

    Supports any provider accessible through LiteLLM, including
    OpenRouter (use 'openrouter/provider/model' strings).

    Each agent instance carries its own api_key so multiple agents
    with different keys can run in the same process without conflict.
    """

    MAX_HISTORY_TICKS = 5
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        agent_id: str,
        model_name: str = "openrouter/qwen/qwq-32b:free",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        super().__init__(agent_id=agent_id, model_name=model_name)

        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. Run: pip install litellm"
            )

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = build_system_prompt()
        self.message_history: list = []

        self.base_url: Optional[str] = (
            self.OPENROUTER_BASE_URL
            if model_name.startswith("openrouter/")
            else None
        )

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the current game state to the LLM and returns the parsed action payload.
        Handles None responses and malformed JSON gracefully.
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

        call_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "timeout": self.timeout,
        }

        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        if self.base_url:
            call_kwargs["base_url"] = self.base_url

        response = litellm.completion(**call_kwargs)

        raw_content = response.choices[0].message.content

        # Guard against None or empty responses from unreliable free-tier models.
        # Return an empty action payload so the tick still advances cleanly.
        if not raw_content or raw_content.strip() == "":
            self.error_log.append(
                f"Tick {state.get('tick', '?')}: Model returned None/empty content."
            )
            return {"thoughts": "Model returned empty response.", "actions": []}

        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0

        self.message_history.append({"role": "assistant", "content": raw_content})

        # Strip markdown code fences if the model ignores json_object format
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        return json.loads(cleaned)

    def reset(self):
        """Resets conversation history and token counters for a fresh benchmark run."""
        self.message_history = []
        self.tick_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.error_log = []
