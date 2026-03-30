"""
run_benchmark.py
----------------
Entry point for the benchmark. Runs one or more LLM agents against the
ai-datacenter-sim environment and prints a leaderboard.

Usage:
    python run_benchmark.py

Setup:
    1. Copy config/.env.example to .env in the project root.
    2. Set OPENROUTER_API_KEY to your key (one key is enough).
    3. pip install -r requirements.txt

To benchmark multiple models, add more entries to the MODELS list below.
Each model automatically uses OPENROUTER_API_KEY unless you add an optional
'api_key_env' field pointing to a different env var.

Output:
    - Leaderboard printed to terminal
    - Summary     -> output/benchmark_<timestamp>.csv
    - Tick logs   -> output/tick_log_<model>_<timestamp>.csv
"""

import os
import litellm
litellm.suppress_debug_info = True

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from benchmark.runner import run_single_agent
from benchmark.leaderboard import print_leaderboard, export_csv, export_tick_logs
from agents.llm_agent import LLMAgent


# ------------------------------------------------------------
# MODEL CONFIGURATION
#
# Add more dicts to this list to benchmark multiple models.
# 'api_key_env' is optional — if omitted or empty, falls back
# to OPENROUTER_API_KEY from .env automatically.
#
# Good free models on OpenRouter (March 2026, all 262K context):
#   openrouter/nvidia/nemotron-3-super-120b-a12b:free  (AI Agents, 262K)
#   openrouter/qwen/qwen3-next-80b-a3b-instruct:free   (Agents/RAG, 262K)
#   openrouter/mistralai/devstral-2512:free            (Coding, 262K)
#   openrouter/mistralai/mistral-small-3.1-24b-instruct:free (General, 32K)
# ------------------------------------------------------------

MODELS = [
    {
        "model": "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    },
]

TICKS = 100
SEED = 42
ENABLE_EVENTS = True
VERBOSE = True
TIMEOUT_SECONDS = 60
MAX_TOKENS = 1024
TICK_DELAY_SECONDS = 8


# ------------------------------------------------------------
# KEY RESOLVER
# ------------------------------------------------------------

def resolve_api_key(entry: dict) -> str | None:
    """Return per-model key if specified, otherwise the shared OPENROUTER_API_KEY."""
    env_var = entry.get("api_key_env")
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key
    return os.getenv("OPENROUTER_API_KEY")


# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

def run_agent_worker(i: int, entry: dict) -> dict | None:
    model_name = entry["model"]
    api_key = resolve_api_key(entry)

    if not api_key:
        print(
            f"  [SKIP] {model_name}: OPENROUTER_API_KEY not set in .env.\n"
            f"  Copy config/.env.example to .env and add your key."
        )
        return None

    total = len(MODELS)
    print(f"\n{'=' * 60}")
    print(f"  Agent {i} of {total}: {model_name}")
    print(f"{'=' * 60}")

    agent = LLMAgent(
        agent_id=f"agent_{i:02d}",
        model_name=model_name,
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
        max_tokens=MAX_TOKENS,
    )

    result = run_single_agent(
        agent=agent,
        ticks=TICKS,
        seed=SEED,
        enable_events=ENABLE_EVENTS,
        verbose=VERBOSE,
        tick_delay=TICK_DELAY_SECONDS,
    )

    print(f"  [{model_name}] Done in {result['elapsed_seconds']}s | Errors: {result['agent_errors']}")
    return result


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  ai-datacenter-sim  |  Benchmark")
    print("=" * 60)
    print(f"  Models     : {len(MODELS)}")
    print(f"  Ticks      : {TICKS}")
    print(f"  Seed       : {SEED}")
    print(f"  Events     : {ENABLE_EVENTS}")
    print(f"  Timeout    : {TIMEOUT_SECONDS}s per call")
    print(f"  Tick delay : {TICK_DELAY_SECONDS}s (rate-limit guard)")
    mode = "PARALLEL" if len(MODELS) > 1 else "SINGLE"
    print(f"  Mode       : {mode}")
    print("=" * 60)

    if not MODELS:
        print("ERROR: No models defined in MODELS list.")
        exit(1)

    results = []

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {
            pool.submit(run_agent_worker, i, entry): entry
            for i, entry in enumerate(MODELS, start=1)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

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
