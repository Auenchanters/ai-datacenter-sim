import json
from typing import Any, Dict
from engine.grid import FacilityGrid, SAFE_TEMP
from engine.hardware import ServerRack, CoolingUnit
from economy.billing import BillingEngine


def serialize_state(
    tick: int,
    grid: FacilityGrid,
    servers: list,
    coolers: list,
    billing: BillingEngine,
    active_jobs: list,
    pending_contracts: list,
) -> Dict[str, Any]:
    """
    Converts the full internal simulation state into a token-efficient
    JSON payload to be sent to the AI agent every tick.

    Design principles:
        - Never send the full raw temperature grid. Only send hotspots.
        - Round all floats to 2-3 decimal places to reduce token count.
        - Include only fields the AI can act upon.
    """
    financial = billing.get_financial_summary()
    power_it = round(sum(s.current_power_kw for s in servers), 3)
    power_cooling = round(sum(c.current_power_kw for c in coolers), 3)
    total_power = round(power_it + power_cooling, 3)
    pue = round(total_power / power_it, 4) if power_it > 0 else 0.0

    payload = {
        "tick": tick,
        "global_metrics": {
            "bank_balance": financial["bank_balance"],
            "net_profit": financial["net_profit"],
            "current_pue": pue,
            "it_power_kw": power_it,
            "cooling_power_kw": power_cooling,
            "total_power_kw": total_power,
            "effective_rate_per_kwh": financial["effective_cost_per_kwh"],
            "price_event_active": financial["price_event_active"],
            "price_event_multiplier": financial["price_event_multiplier"],
            "price_event_ticks_remaining": financial["price_event_ticks_remaining"],
        },
        "facility_grid": {
            "dimensions": {"width": grid.width, "height": grid.height},
            "ambient_temp_c": 22.5,
            "thermal_hotspots": grid.get_hotspots(threshold=SAFE_TEMP),
        },
        "equipment": {
            "servers": [s.to_dict() for s in servers],
            "cooling_units": [c.to_dict() for c in coolers],
        },
        "workload_market": {
            "active_jobs": [
                {
                    "id": job["id"],
                    "type": job["type"],
                    "compute_required": job["compute_required"],
                    "reward_per_tick": job["reward_per_tick"],
                    "ticks_remaining": job["ticks_remaining"],
                    "assigned_racks": job.get("assigned_racks", []),
                }
                for job in active_jobs
            ],
            "pending_contracts": [
                {
                    "id": contract["id"],
                    "type": contract["type"],
                    "display_name": contract["display_name"],
                    "compute_required": contract["compute_required"],
                    "reward_per_tick": contract["reward_per_tick"],
                    "expires_in_ticks": contract["expires_in_ticks"],
                    "description": contract["description"],
                }
                for contract in pending_contracts
            ],
        },
    }

    return payload


def state_to_json_string(state_payload: Dict[str, Any]) -> str:
    return json.dumps(state_payload, indent=2)
