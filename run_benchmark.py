"""
run_benchmark.py
----------------
Entry point for running the full multi-model benchmark suite.

Each model has its own dedicated OpenRouter API key stored as a
separate environment variable. This allows three separate free-tier
keys to be used simultaneously without hitting per-key rate limits.

Usage:
    python run_benchmark.py

Setup:
    1. Copy config/.env.example to .env in the root directory.
    2. Fill in your three OpenRouter API keys.
    3. Install dependencies: pip install -r requirements.txt

Output:
    - Leaderboard printed to terminal.
    - Summary     -> output/benchmark_<timestamp>.csv
    - Tick logs   -> output/tick_log_<model>_<timestamp>.csv
"""

import os
from dotenv import load_dotenv

load_dotenv()

from benchmark.runner import run_single_agent
from benchmark.leaderboard import print_leaderboard, export_csv, export_tick_logs
from agents.llm_agent import LLMAgent


# ------------------------------------------------------------
# MODEL CONFIGURATION
# Each entry maps an OpenRouter model string to the .env
# variable that holds the API key for that model.
# Add or remove entries to change who competes.
# ------------------------------------------------------------

MODELS = [
    {
        "model": "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "api_key_env": "OPENROUTER_API_KEY_NEMOTRON",
    },
    {
        "model": "openrouter/minimax/minimax-m2.5:free",
        "api_key_env": "OPENROUTER_API_KEY_MINIMAX",
    },
    {
        "model": "openrouter/qwen/qwen3-coder:free",
        "api_key_env": "OPENROUTER_API_KEY_QWEN",
    },
]

# Number of simulation ticks each model will play.
# 50 ticks is good for a quick test. 100 ticks is the full benchmark.
TICKS = 100

# Random seed for workload generation.
# Keep this the same across all runs for fair comparison.
SEED = 42

# Whether to inject dynamic challenge events (price spikes, hardware failures).
# Set to False for a clean baseline run with no surprises.
ENABLE_EVENTS = True

# Whether to print per-tick output for each model.
VERBOSE = True


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  ai-datacenter-sim  |  Multi-Model Benchmark")
    print("=" * 60)
    print(f"  Models  : {len(MODELS)}")
    print(f"  Ticks   : {TICKS}")
    print(f"  Seed    : {SEED}")
    print(f"  Events  : {ENABLE_EVENTS}")
    print("=" * 60)

    if not MODELS:
        print("ERROR: No models defined in MODELS list.")
        exit(1)

    results = []

    for i, entry in enumerate(MODELS, start=1):
        model_name = entry["model"]
        api_key_env = entry["api_key_env"]
        api_key = os.getenv(api_key_env)

        if not api_key:
            print(f"  [SKIP] {model_name}: missing env var '{api_key_env}'. Add it to .env and retry.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Agent {i} of {len(MODELS)}: {model_name}")
        print(f"{'=' * 60}")

        agent = LLMAgent(
            agent_id=f"agent_{i:02d}",
            model_name=model_name,
            api_key=api_key,
        )

        result = run_single_agent(
            agent=agent,
            ticks=TICKS,
            seed=SEED,
            enable_events=ENABLE_EVENTS,
            verbose=VERBOSE,
        )

        results.append(result)
        print(f"  Completed in {result['elapsed_seconds']}s")

    if not results:
        print("No agents completed. Check your .env file.")
        exit(1)

    results.sort(key=lambda r: r["final_pue"])

    print_leaderboard(results)

    summary_path = export_csv(results)
    tick_log_paths = export_tick_logs(results)

    print()
    print("Output files:")
    print(f"  Summary  : {summary_path}")
    for p in tick_log_paths:
        print(f"  Tick log : {p}")
    print()
