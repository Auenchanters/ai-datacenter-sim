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
    The key is passed explicitly on every call — litellm global state
    is never used so parallel agents cannot bleed keys into each other.

    NOTE: We intentionally do NOT pass response_format={"type": "json_object"}
    because free-tier models on OpenRouter do not support it and silently
    return None content when it is present. JSON output is enforced via
    the system prompt instead.
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

        # Prefer explicitly passed key; do NOT fall back to a shared global env var
        # so that parallel agents each use their own dedicated key.
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                f"[{agent_id}] No API key provided for model '{model_name}'. "
                f"Pass api_key= explicitly or set the correct env var in .env."
            )

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
        JSON output is enforced via the system prompt. response_format is NOT sent
        because free-tier OpenRouter models silently fail when it is present.
        """
        state_json = json.dumps(state, indent=2)
        user_message = (
            f"TICK {state['tick']} STATE:\n{state_json}\n\n"
            "Respond with ONLY a valid JSON object matching the format in your instructions. "
            "No markdown, no code fences, no explanation text outside the JSON."
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
            # DO NOT add response_format here - it breaks free-tier OpenRouter models
            "timeout": self.timeout,
            # Always pass the key explicitly per-call so parallel agents
            # never share or overwrite each other's credentials via litellm globals.
            "api_key": self.api_key,
        }

        if self.base_url:
            call_kwargs["base_url"] = self.base_url

        response = litellm.completion(**call_kwargs)
        raw_content = response.choices[0].message.content

        # Guard: None or empty content means the model failed silently.
        if not raw_content or raw_content.strip() == "":
            self.error_log.append(
                f"Tick {state.get('tick', '?')}: Model returned None/empty content."
            )
            return {"thoughts": "Model returned empty response.", "actions": []}

        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0

        self.message_history.append({"role": "assistant", "content": raw_content})

        # Strip markdown code fences if the model wraps its output despite instructions.
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        # If the model produced a <think>...</think> block before the JSON
        # (common with QwQ reasoning models), strip it before parsing.
        if "<think>" in cleaned and "</think>" in cleaned:
            think_end = cleaned.rfind("</think>")
            cleaned = cleaned[think_end + len("</think>"):].strip()

        # Find the outermost JSON object in case any stray text still surrounds it.
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start:brace_end + 1]

        return json.loads(cleaned)

    def reset(self):
        """Resets conversation history and token counters for a fresh benchmark run."""
        self.message_history = []
        self.tick_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.error_log = []
