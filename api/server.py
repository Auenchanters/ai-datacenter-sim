import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.grid import FacilityGrid
from engine.tick import SimulationState, step
from economy.billing import BillingEngine
from economy.workloads import WorkloadSpawner
from economy.catalog import HARDWARE_CATALOG
from api.serializer import serialize_state
from api.parser import parse_and_execute


GRID_WIDTH = 20
GRID_HEIGHT = 20
STARTING_BUDGET = 500_000.0
ELECTRICITY_COST = 0.12
MAX_TICKS = 100


class ActionPayload(BaseModel):
    thoughts: str = ""
    actions: list = []


sim_state: Optional[SimulationState] = None
billing: Optional[BillingEngine] = None
spawner: Optional[WorkloadSpawner] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_state, billing, spawner
    grid = FacilityGrid(width=GRID_WIDTH, height=GRID_HEIGHT)
    sim_state = SimulationState(
        grid=grid,
        starting_budget=STARTING_BUDGET,
        electricity_cost_per_kwh=ELECTRICITY_COST,
    )
    billing = BillingEngine(
        starting_budget=STARTING_BUDGET,
        cost_per_kwh=ELECTRICITY_COST,
    )
    spawner = WorkloadSpawner(seed=42)
    yield


app = FastAPI(
    title="ai-datacenter-sim",
    description="LLM benchmark simulation. Agents manage a 2D data center for maximum PUE and profit.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """
    Returns the current serialized game state without advancing the tick.
    Use this to inspect the simulation at any point.
    """
    if sim_state is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized.")

    return serialize_state(
        tick=sim_state.current_tick,
        grid=sim_state.grid,
        servers=sim_state.servers,
        coolers=sim_state.coolers,
        billing=billing,
        active_jobs=sim_state.active_jobs,
        pending_contracts=sim_state.pending_contracts,
    )


@app.post("/tick")
def advance_tick(payload: ActionPayload) -> Dict[str, Any]:
    """
    Accepts the AI's action payload, executes the commands,
    advances the simulation by one tick, spawns new contracts,
    and returns the updated state.
    """
    if sim_state is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized.")

    if sim_state.current_tick >= MAX_TICKS:
        raise HTTPException(status_code=410, detail="Simulation has ended. Check /leaderboard.")

    action_dict = payload.dict()
    parse_results = parse_and_execute(
        action_payload=action_dict,
        grid=sim_state.grid,
        servers=sim_state.servers,
        coolers=sim_state.coolers,
        billing=billing,
        active_jobs=sim_state.active_jobs,
        pending_contracts=sim_state.pending_contracts,
        tick=sim_state.current_tick,
    )

    tick_summary = step(sim_state)

    new_contracts = spawner.spawn(sim_state.current_tick)
    sim_state.pending_contracts.extend(new_contracts)

    billing.tick_price_event()

    new_state = serialize_state(
        tick=sim_state.current_tick,
        grid=sim_state.grid,
        servers=sim_state.servers,
        coolers=sim_state.coolers,
        billing=billing,
        active_jobs=sim_state.active_jobs,
        pending_contracts=sim_state.pending_contracts,
    )

    return {
        "parse_results": parse_results,
        "tick_summary": tick_summary,
        "new_state": new_state,
    }


@app.get("/leaderboard")
def get_leaderboard() -> Dict[str, Any]:
    """
    Returns the final performance scores for the current simulation session.
    In benchmark mode, this endpoint aggregates results across all competing agents.
    """
    if sim_state is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized.")

    financial = billing.get_financial_summary()

    return {
        "ticks_completed": sim_state.current_tick,
        "final_pue": sim_state.current_pue,
        "final_bank_balance": financial["bank_balance"],
        "net_profit": financial["net_profit"],
        "total_revenue": financial["total_revenue"],
        "total_electricity_cost": financial["total_electricity_cost"],
        "total_hardware_spent": financial["total_hardware_spent"],
        "sla_uptime_percent": sim_state.sla_uptime_percent,
        "servers_placed": len(sim_state.servers),
        "coolers_placed": len(sim_state.coolers),
        "jobs_completed": len(sim_state.completed_jobs),
        "jobs_failed": len(sim_state.failed_jobs),
    }


@app.get("/reset")
def reset_simulation() -> Dict[str, str]:
    """
    Resets the simulation to its initial state.
    Use this between benchmark runs to ensure a clean slate for each agent.
    """
    global sim_state, billing, spawner
    grid = FacilityGrid(width=GRID_WIDTH, height=GRID_HEIGHT)
    sim_state = SimulationState(
        grid=grid,
        starting_budget=STARTING_BUDGET,
        electricity_cost_per_kwh=ELECTRICITY_COST,
    )
    billing = BillingEngine(
        starting_budget=STARTING_BUDGET,
        cost_per_kwh=ELECTRICITY_COST,
    )
    spawner = WorkloadSpawner(seed=42)
    return {"status": "Simulation reset successfully."}
