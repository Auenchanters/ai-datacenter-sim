import random
import uuid
from typing import List, Dict, Any, Optional


JOB_TYPES = [
    {
        "type": "AI_TRAINING",
        "display_name": "AI Model Training Job",
        "compute_required": 1000,
        "reward_per_tick": 5000.0,
        "min_expiry_ticks": 8,
        "max_expiry_ticks": 15,
        "description": (
            "High-value contract. Requires large GPU compute capacity. "
            "Expires quickly if not accepted. Best ROI per compute unit."
        ),
    },
    {
        "type": "AI_INFERENCE",
        "display_name": "AI Inference Serving Job",
        "compute_required": 500,
        "reward_per_tick": 2500.0,
        "min_expiry_ticks": 10,
        "max_expiry_ticks": 20,
        "description": (
            "Medium-value contract. Requires GPU or high-end CPU capacity. "
            "Steady revenue stream. Moderate expiry window."
        ),
    },
    {
        "type": "WEB_HOSTING",
        "display_name": "Web Hosting and CDN Job",
        "compute_required": 100,
        "reward_per_tick": 400.0,
        "min_expiry_ticks": 15,
        "max_expiry_ticks": 30,
        "description": (
            "Low-value contract. Suitable for basic CPU racks. "
            "Long expiry window gives the AI time to provision resources. "
            "Low risk, low reward."
        ),
    },
    {
        "type": "DATABASE_HOSTING",
        "display_name": "Enterprise Database Hosting",
        "compute_required": 200,
        "reward_per_tick": 900.0,
        "min_expiry_ticks": 12,
        "max_expiry_ticks": 25,
        "description": (
            "Moderate-value contract. Requires reliable CPU racks with "
            "stable uptime. Penalizes SLA failures more harshly than "
            "other job types."
        ),
    },
    {
        "type": "HPC_SIMULATION",
        "display_name": "High-Performance Computing Simulation",
        "compute_required": 800,
        "reward_per_tick": 3800.0,
        "min_expiry_ticks": 6,
        "max_expiry_ticks": 12,
        "description": (
            "High-value, time-sensitive contract. Requires significant "
            "compute density. Very short expiry window tests the AI's "
            "ability to provision hardware proactively."
        ),
    },
]

JOB_TYPE_MAP: Dict[str, Dict[str, Any]] = {j["type"]: j for j in JOB_TYPES}


class WorkloadSpawner:
    """
    Generates dynamic compute contracts each simulation tick.

    The spawner uses a weighted probability system so that early in
    the game, simpler low-compute jobs are more likely, giving the AI
    time to provision hardware before demanding GPU workloads appear.

    After a configurable tick threshold, the mix shifts toward
    high-value AI training and HPC jobs.
    """

    EARLY_WEIGHTS = [0.10, 0.15, 0.40, 0.30, 0.05]
    LATE_WEIGHTS  = [0.30, 0.25, 0.15, 0.15, 0.15]
    LATE_GAME_TICK = 20

    def __init__(
        self,
        min_contracts_per_tick: int = 1,
        max_contracts_per_tick: int = 3,
        seed: Optional[int] = None,
    ):
        self.min_contracts_per_tick = min_contracts_per_tick
        self.max_contracts_per_tick = max_contracts_per_tick
        self.rng = random.Random(seed)

    def spawn(self, current_tick: int) -> List[Dict[str, Any]]:
        """
        Generates a batch of new contracts for the current tick.
        Each contract is assigned a unique job_id and an expiry counter.
        """
        weights = (
            self.LATE_WEIGHTS
            if current_tick >= self.LATE_GAME_TICK
            else self.EARLY_WEIGHTS
        )

        count = self.rng.randint(
            self.min_contracts_per_tick, self.max_contracts_per_tick
        )

        contracts = []
        for _ in range(count):
            template = self.rng.choices(JOB_TYPES, weights=weights, k=1)[0]
            expiry = self.rng.randint(
                template["min_expiry_ticks"], template["max_expiry_ticks"]
            )
            contract = {
                "id": f"job_{uuid.uuid4().hex[:8]}",
                "type": template["type"],
                "display_name": template["display_name"],
                "compute_required": template["compute_required"],
                "reward_per_tick": template["reward_per_tick"],
                "expires_in_ticks": expiry,
                "assigned_racks": [],
                "ticks_remaining": expiry,
                "description": template["description"],
            }
            contracts.append(contract)

        return contracts

    def get_all_job_types_for_prompt(self) -> List[Dict[str, Any]]:
        """
        Returns a minimal list of job type definitions for the AI system prompt.
        Gives the AI context on what kinds of workloads it should prepare hardware for.
        """
        return [
            {
                "type": j["type"],
                "compute_required": j["compute_required"],
                "reward_per_tick": j["reward_per_tick"],
                "description": j["description"],
            }
            for j in JOB_TYPES
        ]
