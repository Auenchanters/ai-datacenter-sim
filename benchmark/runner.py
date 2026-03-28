import json
import time
from typing import Any, Dict, List, Optional

from engine.grid import FacilityGrid
from engine.tick import SimulationState, step
from economy.billing import BillingEngine
from economy.workloads import WorkloadSpawner
from api.serializer import serialize_state
from api.parser import parse_and_execute
from benchmark.events import get_standard_event_schedule, BenchmarkEvent


def run_single_agent(
    agent,
    ticks: int = 100,
    seed: int = 42,
    grid_width: int = 20,
    grid_height: int = 20,
    starting_budget: float = 500_000.0,
    electricity_cost: float = 0.12,
    enable_events: bool = True,
    verbose: bool = True,
    tick_delay: float = 0.0,
) -> Dict[str, Any]:
    """
    Runs a complete simulation for a single agent and returns its final score.
    Uses a fixed seed so results are reproducible and comparable across agents.

    tick_delay: seconds to sleep after each tick's LLM call, used to stay
                under OpenRouter's free-tier rate limit (8 RPM per model).
                Set to 8.0 when running multiple parallel agents on free tier.
    """
    grid    = FacilityGrid(width=grid_width, height=grid_height)
    sim     = SimulationState(grid=grid, starting_budget=starting_budget, electricity_cost_per_kwh=electricity_cost)
    billing = BillingEngine(starting_budget=starting_budget, cost_per_kwh=electricity_cost)
    spawner = WorkloadSpawner(seed=seed)
    events  = get_standard_event_schedule() if enable_events else []

    tick_log = []
    start_time = time.time()

    for tick_num in range(ticks):
        new_contracts = spawner.spawn(tick_num)
        sim.pending_contracts.extend(new_contracts)

        for event in events:
            event.try_trigger(tick_num, sim, billing)

        state = serialize_state(
            tick=tick_num,
            grid=grid,
            servers=sim.servers,
            coolers=sim.coolers,
            billing=billing,
            active_jobs=sim.active_jobs,
            pending_contracts=sim.pending_contracts,
        )

        action_payload = agent.safe_decide(state)

        # Rate-limit guard: pause after each LLM call so parallel agents
        # don't exceed OpenRouter's free-tier 8 RPM cap.
        if tick_delay > 0:
            time.sleep(tick_delay)

        parse_results = parse_and_execute(
            action_payload=action_payload,
            grid=grid,
            servers=sim.servers,
            coolers=sim.coolers,
            billing=billing,
            active_jobs=sim.active_jobs,
            pending_contracts=sim.pending_contracts,
            tick=tick_num,
        )

        tick_summary = step(sim)
        billing.tick_price_event()

        for event in events:
            event.tick_down(sim, billing)

        tick_log.append({
            "tick": tick_num,
            "pue": tick_summary["pue"],
            "bank_balance": tick_summary["bank_balance"],
            "revenue": tick_summary["revenue_this_tick"],
            "electricity_cost": tick_summary["electricity_cost_this_tick"],
            "sla_uptime": tick_summary["sla_uptime_percent"],
        })

        if verbose:
            print(
                f"[{agent.model_name}] Tick {tick_num:03d} | "
                f"PUE: {tick_summary['pue']:.3f} | "
                f"Bal: ${tick_summary['bank_balance']:>12,.0f} | "
                f"SLA: {tick_summary['sla_uptime_percent']:.1f}%"
            )

    elapsed = round(time.time() - start_time, 2)
    financial = billing.get_financial_summary()

    return {
        "agent_id": agent.agent_id,
        "model_name": agent.model_name,
        "ticks_completed": ticks,
        "elapsed_seconds": elapsed,
        "final_pue": sim.current_pue,
        "final_bank_balance": financial["bank_balance"],
        "net_profit": financial["net_profit"],
        "total_revenue": financial["total_revenue"],
        "total_electricity_cost": financial["total_electricity_cost"],
        "total_hardware_spent": financial["total_hardware_spent"],
        "sla_uptime_percent": sim.sla_uptime_percent,
        "jobs_completed": len(sim.completed_jobs),
        "jobs_failed": len(sim.failed_jobs),
        "servers_placed": len(sim.servers),
        "coolers_placed": len(sim.coolers),
        "total_tokens_used": agent.total_prompt_tokens + agent.total_completion_tokens,
        "agent_errors": len(agent.error_log),
        "tick_log": tick_log,
    }


def run_benchmark(
    model_names: List[str],
    ticks: int = 100,
    seed: int = 42,
    enable_events: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Runs the full benchmark suite across all specified model names.
    Each model gets an identical simulation: same seed, same events, same budget.
    Returns a list of result dictionaries sorted by final PUE (ascending = better).
    """
    from agents.llm_agent import LLMAgent

    all_results = []

    for i, model_name in enumerate(model_names):
        print(f"\n{'=' * 60}")
        print(f"Running agent {i + 1} of {len(model_names)}: {model_name}")
        print(f"{'=' * 60}")

        agent = LLMAgent(
            agent_id=f"agent_{i + 1:02d}",
            model_name=model_name,
        )

        result = run_single_agent(
            agent=agent,
            ticks=ticks,
            seed=seed,
            enable_events=enable_events,
            verbose=verbose,
        )

        all_results.append(result)
        print(f"Completed {model_name} in {result['elapsed_seconds']}s")

    all_results.sort(key=lambda r: r["final_pue"])
    return all_results
