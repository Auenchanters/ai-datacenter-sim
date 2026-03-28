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
    1. Copy config/.env.example to .env in the project root.
    2. Fill in your three OpenRouter API keys.
    3. pip install -r requirements.txt

Output:
    - Leaderboard printed to terminal.
    - Summary     -> output/benchmark_<timestamp>.csv
    - Tick logs   -> output/tick_log_<model>_<timestamp>.csv
"""

import os
import litellm
litellm.suppress_debug_info = True

from dotenv import load_dotenv
load_dotenv()

from benchmark.runner import run_single_agent
from benchmark.leaderboard import print_leaderboard, export_csv, export_tick_logs
from agents.llm_agent import LLMAgent


# ------------------------------------------------------------
# MODEL CONFIGURATION
#
# Reliable free models on OpenRouter as of March 2026:
#   openrouter/qwen/qwq-32b:free            - strong reasoning
#   openrouter/deepseek/deepseek-r1:free    - strong reasoning
#   openrouter/qwen/qwen3-coder:free        - good for structured JSON
#   openrouter/minimax/minimax-m2.5:free    - fast responses
#
# Each entry maps a model string to the .env variable
# that holds the API key for that model.
# ------------------------------------------------------------

MODELS = [
    {
        "model": "openrouter/qwen/qwq-32b:free",
        "api_key_env": "OPENROUTER_API_KEY_NEMOTRON",  # reuse existing key slot
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

TICKS = 100
SEED = 42
ENABLE_EVENTS = True
VERBOSE = True
TIMEOUT_SECONDS = 60  # per LLM call; free models can be slow


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
    print(f"  Timeout : {TIMEOUT_SECONDS}s per call")
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
            timeout=TIMEOUT_SECONDS,
        )

        result = run_single_agent(
            agent=agent,
            ticks=TICKS,
            seed=SEED,
            enable_events=ENABLE_EVENTS,
            verbose=VERBOSE,
        )

        results.append(result)
        print(f"  Completed in {result['elapsed_seconds']}s | Errors: {result['agent_errors']}")

    if not results:
        print("\nNo agents completed. Check your .env file.")
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
