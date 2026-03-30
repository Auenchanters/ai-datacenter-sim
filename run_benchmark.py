"""
run_benchmark.py
----------------
Entry point for the benchmark. Runs one or more LLM agents in PARALLEL against
the ai-datacenter-sim environment and prints a leaderboard.

Usage:
    python run_benchmark.py

Setup:
    1. Copy config/.env.example to .env in the project root.
    2. Set GROQ_API_KEY and/or GEMINI_API_KEY in .env
    3. pip install -r requirements.txt

FREE API KEYS:
    Groq  : https://console.groq.com          (~14,400 req/day)
    Gemini: https://aistudio.google.com       (~1,500 req/day)

Both agents run in parallel on the same 50-tick scenario.
Tick delays are staggered so they don't fire at the exact same millisecond
and eat into each other's per-minute rate limits.

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
# Both run in parallel. Comment out either to run solo.
# api_key_env: which .env variable holds the key for this model
# tick_delay:  seconds between ticks (rate-limit guard per model)
# ------------------------------------------------------------

MODELS = [
    {
        "model": "groq/llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "tick_delay": 2.0,   # Groq: fast, generous quota
    },
    {
        "model": "gemini/gemini-2.0-flash",
        "api_key_env": "GEMINI_API_KEY",
        "tick_delay": 3.0,   # Gemini: slightly slower, 15 RPM free tier
    },
]

TICKS = 50             # 50 ticks: enough depth, stays well under daily quotas
SEED  = 42             # Fixed seed = same events for both agents (fair comparison)
ENABLE_EVENTS  = True
VERBOSE        = True
TIMEOUT_SECONDS = 60
MAX_TOKENS      = 1024


# ------------------------------------------------------------
# KEY RESOLVER
# ------------------------------------------------------------

def resolve_api_key(entry: dict) -> str | None:
    env_var = entry.get("api_key_env")
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key
    # Fallback chain
    return (
        os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )


# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

def run_agent_worker(i: int, entry: dict) -> dict | None:
    model_name  = entry["model"]
    api_key     = resolve_api_key(entry)
    tick_delay  = entry.get("tick_delay", 2.0)

    if not api_key:
        print(
            f"  [SKIP] {model_name}: No API key found.\n"
            f"  Set {entry.get('api_key_env', 'GROQ_API_KEY or GEMINI_API_KEY')} in .env"
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
        tick_delay=tick_delay,
    )

    print(f"  [{model_name}] Done in {result['elapsed_seconds']}s | Errors: {result['agent_errors']}")
    return result


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    # Detect which models have keys available
    available = []
    for entry in MODELS:
        if resolve_api_key(entry):
            available.append(entry)
        else:
            print(f"  [SKIP] {entry['model']}: {entry.get('api_key_env')} not set in .env")

    if not available:
        print("\nERROR: No API keys found. Set GROQ_API_KEY and/or GEMINI_API_KEY in .env")
        exit(1)

    print()
    print("=" * 60)
    print("  ai-datacenter-sim  |  Benchmark")
    print("=" * 60)
    print(f"  Models     : {len(available)} active")
    print(f"  Ticks      : {TICKS}")
    print(f"  Seed       : {SEED}")
    print(f"  Events     : EXTREME (cascading failures enabled)")
    print(f"  Timeout    : {TIMEOUT_SECONDS}s per call")
    mode = "PARALLEL" if len(available) > 1 else "SINGLE"
    print(f"  Mode       : {mode}")
    for e in available:
        print(f"    - {e['model']} (delay: {e.get('tick_delay', 2.0)}s/tick)")
    print("=" * 60)

    results = []

    with ThreadPoolExecutor(max_workers=len(available)) as pool:
        futures = {
            pool.submit(run_agent_worker, i, entry): entry
            for i, entry in enumerate(available, start=1)
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

    summary_path   = export_csv(results)
    tick_log_paths = export_tick_logs(results)

    print()
    print("Output files:")
    print(f"  Summary  : {summary_path}")
    for p in tick_log_paths:
        print(f"  Tick log : {p}")
    print()
