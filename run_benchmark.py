"""
run_benchmark.py
----------------
Entry point for running the full multi-model benchmark suite.

Usage:
    python run_benchmark.py

Before running:
    1. Copy config/.env.example to .env in the root directory.
    2. Fill in your API keys for the providers you want to test.
    3. Install dependencies: pip install -r requirements.txt

To test with free local models (no API key required):
    1. Install Ollama from https://ollama.com
    2. Run: ollama pull llama3
    3. Add "ollama/llama3" to the MODELS list below.

Output:
    - Leaderboard printed to terminal.
    - Summary CSV saved to output/benchmark_<timestamp>.csv
    - Per-tick logs saved to output/tick_log_<model>_<timestamp>.csv
"""

import os
from dotenv import load_dotenv

load_dotenv()

from benchmark.runner import run_benchmark
from benchmark.leaderboard import print_leaderboard, export_csv, export_tick_logs


# ------------------------------------------------------------
# CONFIGURATION
# Add or remove model strings to change who competes.
# All models run against the same seed and event schedule.
# Full list of supported model strings:
# https://docs.litellm.ai/docs/providers
# ------------------------------------------------------------

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku-20240307",
    # "openai/gpt-4o",
    # "anthropic/claude-3-5-sonnet-20241022",
    # "gemini/gemini-1.5-pro",
    # "ollama/llama3",
    # "ollama/mistral",
]

# Number of simulation ticks each model will play.
# 50 ticks is good for a quick test. 100 ticks is the full benchmark.
TICKS = 100

# Random seed for workload generation.
# Keep this the same across all runs to ensure fair comparison.
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
        print("ERROR: No models defined. Add at least one model to the MODELS list.")
        exit(1)

    results = run_benchmark(
        model_names=MODELS,
        ticks=TICKS,
        seed=SEED,
        enable_events=ENABLE_EVENTS,
        verbose=VERBOSE,
    )

    print_leaderboard(results)

    summary_path = export_csv(results)
    tick_log_paths = export_tick_logs(results)

    print()
    print("Files saved:")
    print(f"  Summary  : {summary_path}")
    for p in tick_log_paths:
        print(f"  Tick log : {p}")
    print()
