import json
import os
import re
from typing import Any, Dict, Optional

try:
    import litellm
    litellm.suppress_debug_info = True
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from agents.base_agent import BaseAgent
from agents.system_prompt import build_system_prompt


def _repair_json(text: str) -> str:
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _slim_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strips the full simulation state down to a token-minimal summary.
    Target: < 300 tokens per tick (was ~1,800 with full grid + indent=2).
    Removes: full grid dimensions, ambient temp, full server objects.
    Keeps: everything the agent needs to make a decision.
    """
    gm = state.get("global_metrics", {})
    equip = state.get("equipment", {})
    wm = state.get("workload_market", {})
    grid = state.get("facility_grid", {})

    # Slim server list: only the fields the agent acts on
    servers = [
        {
            "id": s.get("instance_id", s.get("id", "?")),
            "status": s.get("status", "?"),
            "compute": s.get("compute_capacity", 0),
            "util": round(s.get("utilization", 0), 2),
            "temp": round(s.get("intake_temp_c", 0), 1),
            "safe_temp": s.get("safe_temp_c", 75),
            "pos": s.get("position", {}),
            "facing": s.get("facing", "?"),
        }
        for s in equip.get("servers", [])
    ]

    # Slim cooler list
    coolers = [
        {
            "id": c.get("instance_id", c.get("id", "?")),
            "status": c.get("status", "?"),
            "fan_speed": round(c.get("fan_speed", 0), 2),
            "pos": c.get("position", {}),
        }
        for c in equip.get("cooling_units", [])
    ]

    # Cap lists to avoid token blowup
    pending = wm.get("pending_contracts", [])[:3]
    active  = wm.get("active_jobs",       [])[:5]

    slim_pending = [
        {
            "id": c["id"],
            "type": c["type"],
            "compute": c["compute_required"],
            "reward": c["reward_per_tick"],
            "expires": c["expires_in_ticks"],
        }
        for c in pending
    ]

    slim_active = [
        {
            "id": j["id"],
            "type": j["type"],
            "compute": j["compute_required"],
            "reward": j["reward_per_tick"],
            "ticks_left": j["ticks_remaining"],
            "racks": j.get("assigned_racks", []),
        }
        for j in active
    ]

    # Collect active event names so the agent can distinguish
    # opportunity events (e.g. COMPUTE_DEMAND_TSUNAMI) from cost events.
    active_events = [
        e.get("type", e.get("name", str(e)))
        for e in state.get("active_events", [])
        if isinstance(e, dict)
    ]
    # Fallback: some engines store events as plain strings
    if not active_events:
        active_events = [
            e for e in state.get("active_events", [])
            if isinstance(e, str)
        ]

    return {
        "tick": state["tick"],
        "balance": gm.get("bank_balance", 0),
        "pue": round(gm.get("current_pue", 0), 3),
        "it_kw": round(gm.get("it_power_kw", 0), 2),
        "cool_kw": round(gm.get("cooling_power_kw", 0), 2),
        "price_mult": gm.get("price_event_multiplier", 1.0),
        "price_ticks_left": gm.get("price_event_ticks_remaining", 0),
        "active_events": active_events,
        "hotspots": grid.get("thermal_hotspots", []),
        "servers": servers,
        "coolers": coolers,
        "pending_contracts": slim_pending,
        "active_jobs": slim_active,
    }


class LLMAgent(BaseAgent):
    """
    LLM-powered simulation agent via LiteLLM.
    Stateless per tick (no conversation history) to minimise token spend.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        agent_id: str,
        model_name: str = "groq/llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        super().__init__(agent_id=agent_id, model_name=model_name)

        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is not installed. Run: pip install litellm")

        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                f"[{agent_id}] No API key provided for model '{model_name}'."
            )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = build_system_prompt()

        self.base_url: Optional[str] = (
            self.OPENROUTER_BASE_URL
            if model_name.startswith("openrouter/")
            else None
        )

        # Token tracking
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        slim = _slim_state(state)
        state_json = json.dumps(slim, separators=(",", ":"))  # compact, no indent

        user_message = (
            f"TICK {slim['tick']}|BAL:{slim['balance']}|PUE:{slim['pue']}|"
            f"PRICE:{slim['price_mult']}x({slim['price_ticks_left']}tks)|"
            f"EVENTS:{','.join(slim['active_events']) if slim['active_events'] else 'none'}\n"
            f"{state_json}\n"
            "Reply ONLY valid JSON. No markdown, no fences, no trailing commas."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_message},
        ]

        call_kwargs = {
            "model":       self.model_name,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "timeout":     self.timeout,
            "api_key":     self.api_key,
        }
        if self.base_url:
            call_kwargs["base_url"] = self.base_url

        response    = litellm.completion(**call_kwargs)
        raw_content = response.choices[0].message.content

        if hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens     += response.usage.prompt_tokens or 0
            self.total_completion_tokens += response.usage.completion_tokens or 0

        if not raw_content or not raw_content.strip():
            return {"thoughts": "Empty response.", "actions": []}

        cleaned = raw_content.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                l for l in cleaned.splitlines() if not l.startswith("```")
            ).strip()

        # Strip <think> blocks
        if "<think>" in cleaned and "</think>" in cleaned:
            cleaned = cleaned[cleaned.rfind("</think>") + len("</think>"):].strip()

        # Extract outermost JSON object
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            cleaned = cleaned[s:e + 1]

        return json.loads(_repair_json(cleaned))

    def reset(self):
        self.tick_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.error_log = []
