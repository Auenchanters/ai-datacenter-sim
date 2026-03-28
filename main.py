import argparse
import json
import os
import sys


def run_server():
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)


def run_headless(model_name: str, ticks: int, seed: int):
    """
    Runs a complete headless simulation loop connecting a single LLM agent
    to the game engine without starting the FastAPI server.
    Useful for quick local benchmarking and testing.
    """
    from engine.grid import FacilityGrid
    from engine.tick import SimulationState, step
    from economy.billing import BillingEngine
    from economy.workloads import WorkloadSpawner
    from api.serializer import serialize_state
    from api.parser import parse_and_execute
    from agents.llm_agent import LLMAgent

    print(f"\nStarting headless simulation.")
    print(f"Model : {model_name}")
    print(f"Ticks : {ticks}")
    print(f"Seed  : {seed}")
    print("-" * 50)

    grid = FacilityGrid(width=20, height=20)
    sim = SimulationState(grid=grid, starting_budget=500_000.0)
    billing = BillingEngine(starting_budget=500_000.0)
    spawner = WorkloadSpawner(seed=seed)
    agent = LLMAgent(agent_id="agent_01", model_name=model_name)

    for tick_num in range(ticks):
        new_contracts = spawner.spawn(tick_num)
        sim.pending_contracts.extend(new_contracts)

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

        print(
            f"Tick {tick_num:03d} | "
            f"PUE: {tick_summary['pue']:.3f} | "
            f"Balance: ${tick_summary['bank_balance']:>12,.2f} | "
            f"Revenue: ${tick_summary['revenue_this_tick']:>8,.2f} | "
            f"Power: ${tick_summary['electricity_cost_this_tick']:>8,.2f} | "
            f"SLA: {tick_summary['sla_uptime_percent']:.1f}%"
        )

        errors = [r for r in parse_results.get("results", []) if r["status"] == "ERROR"]
        if errors:
            for err in errors:
                print(f"  [PARSE ERROR] {err['command']}: {err['message']}")

    print("-" * 50)
    print("SIMULATION COMPLETE")
    financial = billing.get_financial_summary()
    print(f"Final PUE        : {sim.current_pue:.4f}")
    print(f"Final Balance    : ${financial['bank_balance']:,.2f}")
    print(f"Net Profit       : ${financial['net_profit']:,.2f}")
    print(f"SLA Uptime       : {sim.sla_uptime_percent:.2f}%")
    print(f"Jobs Completed   : {len(sim.completed_jobs)}")
    print(f"Jobs Failed      : {len(sim.failed_jobs)}")
    print(f"Servers Placed   : {len(sim.servers)}")
    print(f"Coolers Placed   : {len(sim.coolers)}")
    print(f"Tokens Used      : {agent.total_prompt_tokens + agent.total_completion_tokens:,}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ai-datacenter-sim: LLM benchmark simulation."
    )
    parser.add_argument(
        "--mode",
        choices=["server", "headless"],
        default="server",
        help="Run as FastAPI server (default) or headless loop.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="LiteLLM model string. E.g. openai/gpt-4o-mini, anthropic/claude-3-haiku-20240307, ollama/llama3",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=50,
        help="Number of simulation ticks to run in headless mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for workload generation. Use the same seed across models for fair comparison.",
    )

    args = parser.parse_args()

    if args.mode == "server":
        run_server()
    elif args.mode == "headless":
        run_headless(model_name=args.model, ticks=args.ticks, seed=args.seed)
