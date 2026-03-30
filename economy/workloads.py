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
            "Long expiry window gives the AI time to provision resources. Low risk, low reward."
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
            "stable uptime. Penalizes SLA failures more harshly than other job types."
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
            "compute density. Very short expiry window — provision hardware proactively."
        ),
    },
    {
        "type": "EMERGENCY_FAILOVER",
        "display_name": "Emergency Failover Hosting",
        "compute_required": 600,
        "reward_per_tick": 8000.0,
        "min_expiry_ticks": 3,
        "max_expiry_ticks": 5,
        "description": (
            "CRISIS CONTRACT: Another datacenter has gone offline. "
            "Enormous reward but expires in 3-5 ticks. "
            "Accept and route IMMEDIATELY or the contract vanishes."
        ),
    },
    {
        "type": "RANSOMWARE_RECOVERY",
        "display_name": "Ransomware Recovery Compute",
        "compute_required": 400,
        "reward_per_tick": 6000.0,
        "min_expiry_ticks": 4,
        "max_expiry_ticks": 7,
        "description": (
            "CRISIS CONTRACT: Security firm needs isolated compute for malware analysis. "
            "Very high reward, very short window. "
            "Do NOT assign to any server marked ISOLATED."
        ),
    },
]

JOB_TYPE_MAP: Dict[str, Dict[str, Any]] = {j["type"]: j for j in JOB_TYPES}


class WorkloadSpawner:
    """
    Generates dynamic compute contracts each simulation tick.

    Early game: simple low-compute jobs to let the AI provision hardware.
    Late game: high-value, time-sensitive jobs and crisis contracts.
    Crisis contracts (EMERGENCY_FAILOVER, RANSOMWARE_RECOVERY) spawn
    rarely but at extreme reward — the AI must react within 3-5 ticks.
    """

    # Weights match JOB_TYPES order:
    # AI_TRAINING, AI_INFERENCE, WEB_HOSTING, DATABASE_HOSTING, HPC_SIMULATION,
    # EMERGENCY_FAILOVER, RANSOMWARE_RECOVERY
    EARLY_WEIGHTS = [0.08, 0.12, 0.40, 0.30, 0.05, 0.03, 0.02]
    LATE_WEIGHTS  = [0.25, 0.20, 0.12, 0.12, 0.15, 0.10, 0.06]
    LATE_GAME_TICK = 15  # Crisis contracts start appearing earlier in 50-tick game

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
        return [
            {
                "type": j["type"],
                "compute_required": j["compute_required"],
                "reward_per_tick": j["reward_per_tick"],
                "description": j["description"],
            }
            for j in JOB_TYPES
        ]
