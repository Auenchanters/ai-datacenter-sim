import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List


def print_leaderboard(results: List[Dict[str, Any]]):
    """
    Prints a formatted leaderboard table to stdout.
    Agents are ranked by final PUE (ascending). Lower PUE is better.
    """
    col_w = [
        4,   # rank
        34,  # model name
        8,   # PUE
        14,  # net profit
        10,  # SLA
        8,   # jobs done
        8,   # jobs failed
        10,  # tokens
    ]

    headers = ["Rank", "Model", "PUE", "Net Profit", "SLA %", "Done", "Failed", "Tokens"]

    def row_str(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_w))

    separator = "-" * sum(col_w + [2] * len(col_w))

    print()
    print("=" * len(separator))
    print("  AI DATACENTER SIM  |  BENCHMARK LEADERBOARD")
    print("=" * len(separator))
    print(row_str(headers))
    print(separator)

    for rank, result in enumerate(results, start=1):
        pue_str    = f"{result['final_pue']:.4f}"
        profit_str = f"${result['net_profit']:,.0f}"
        sla_str    = f"{result['sla_uptime_percent']:.1f}"
        tokens_str = f"{result['total_tokens_used']:,}"

        print(row_str([
            rank,
            result["model_name"][:34],
            pue_str,
            profit_str,
            sla_str,
            result["jobs_completed"],
            result["jobs_failed"],
            tokens_str,
        ]))

    print(separator)
    print()


def export_csv(results: List[Dict[str, Any]], output_dir: str = "output"):
    """
    Exports the leaderboard results to a timestamped CSV file.
    The tick_log is excluded from the CSV to keep it concise.
    One row per agent.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = os.path.join(output_dir, f"benchmark_{timestamp}.csv")

    exclude_keys = {"tick_log"}
    fieldnames = [k for k in results[0].keys() if k not in exclude_keys]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {k: v for k, v in result.items() if k not in exclude_keys}
            writer.writerow(row)

    print(f"Leaderboard exported to: {filepath}")
    return filepath


def export_tick_logs(results: List[Dict[str, Any]], output_dir: str = "output"):
    """
    Exports per-tick metric logs for each agent to individual CSV files.
    Useful for plotting PUE and balance over time to compare agent strategies.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []

    for result in results:
        model_slug = result["model_name"].replace("/", "_").replace(".", "_")
        filepath   = os.path.join(output_dir, f"tick_log_{model_slug}_{timestamp}.csv")

        tick_log = result.get("tick_log", [])
        if not tick_log:
            continue

        fieldnames = list(tick_log[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tick_log)

        paths.append(filepath)
        print(f"Tick log exported: {filepath}")

    return paths
