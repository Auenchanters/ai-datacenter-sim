"""
run_benchmark.py
----------------
Entry point for the benchmark. Runs one or more LLM agents against the
ai-datacenter-sim environment and prints a leaderboard.

Usage:
    python run_benchmark.py

Setup:
    1. Copy config/.env.example to .env in the project root.
    2. Set GROQ_API_KEY (recommended) or OPENROUTER_API_KEY.
    3. pip install -r requirements.txt

RECOMMENDED FREE MODELS (March 2026):

  Groq  — fastest, ~14,400 req/day free:
    groq/llama-3.3-70b-versatile      (best JSON quality)
    groq/llama-3.1-8b-instant         (faster, lower quality)
    groq/gemma2-9b-it                 (good JSON, very fast)

  Gemini — 1,500 req/day free:
    gemini/gemini-2.0-flash           (use GEMINI_API_KEY)

  OpenRouter free tier (50-1000 req/day depending on model):
    openrouter/google/gemma-3-27b-it:free
    openrouter/mistralai/mistral-small-3.1-24b-instruct:free

To benchmark multiple models, add more dicts to the MODELS list below.

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
# GROQ (recommended): set GROQ_API_KEY in .env
#   Get a free key at https://console.groq.com
#   ~14,400 free requests/day, very fast, great JSON compliance
#
# GEMINI: set GEMINI_API_KEY in .env
#   Get a free key at https://aistudio.google.com
#   1,500 free requests/day
#
# OPENROUTER: set OPENROUTER_API_KEY in .env
#   Get a free key at https://openrouter.ai
#   50-1000 req/day depending on model (free tier)
# ------------------------------------------------------------

MODELS = [
    {
        "model": "groq/llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    # Uncomment to benchmark multiple models:
    # {"model": "gemini/gemini-2.0-flash", "api_key_env": "GEMINI_API_KEY"},
    # {"model": "openrouter/google/gemma-3-27b-it:free", "api_key_env": "OPENROUTER_API_KEY"},
]

TICKS = 100            # Groq free tier supports this easily (~14,400 req/day)
SEED = 42
ENABLE_EVENTS = True
VERBOSE = True
TIMEOUT_SECONDS = 60
MAX_TOKENS = 1024
TICK_DELAY_SECONDS = 2  # Groq is fast; 2s is enough to stay under per-minute limits


# ------------------------------------------------------------
# KEY RESOLVER
# ------------------------------------------------------------

def resolve_api_key(entry: dict) -> str | None:
    """Return per-model key if specified, otherwise fall back to common keys."""
    env_var = entry.get("api_key_env")
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key
    # Fallback chain: Groq -> Gemini -> OpenRouter
    return (
        os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )


# ------------------------------------------------------------
# WORKER
# ------------------------------------------------------------

def run_agent_worker(i: int, entry: dict) -> dict | None:
    model_name = entry["model"]
    api_key = resolve_api_key(entry)

    if not api_key:
        print(
            f"  [SKIP] {model_name}: No API key found.\n"
            f"  Set GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in .env"
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
    print(f"  Tick delay : {TICK_DELAY_SECONDS}s")
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
