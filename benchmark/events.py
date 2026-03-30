import random
from typing import Any, Callable, Dict, List, Optional


class BenchmarkEvent:
    """
    Defines a single dynamic challenge event injected into the simulation.
    Events are extreme by design — the AI must actively respond or face
    cascading failures, runaway costs, or SLA collapse.
    """

    def __init__(
        self,
        name: str,
        description: str,
        trigger_tick: Optional[int],
        duration_ticks: int,
        apply_fn: Callable,
        revert_fn: Optional[Callable] = None,
    ):
        self.name = name
        self.description = description
        self.trigger_tick = trigger_tick
        self.duration_ticks = duration_ticks
        self.apply_fn = apply_fn
        self.revert_fn = revert_fn
        self.active = False
        self.ticks_remaining = 0
        self.fired = False

    def try_trigger(self, tick: int, sim_state, billing) -> bool:
        if self.fired:
            return False
        if self.trigger_tick is not None and tick == self.trigger_tick:
            self._fire(sim_state, billing)
            return True
        return False

    def _fire(self, sim_state, billing):
        self.apply_fn(sim_state, billing)
        self.active = True
        self.ticks_remaining = self.duration_ticks
        self.fired = True
        print(f"  [EVENT] {self.name}: {self.description}")

    def tick_down(self, sim_state, billing):
        if self.active:
            self.ticks_remaining -= 1
            if self.ticks_remaining <= 0:
                self.active = False
                if self.revert_fn:
                    self.revert_fn(sim_state, billing)
                print(f"  [EVENT END] {self.name} has expired.")


# ---------------------------------------------------------------------------
# EVENT FACTORIES
# ---------------------------------------------------------------------------

def make_electricity_spike(trigger_tick: int, multiplier: float = 3.5, duration: int = 6) -> BenchmarkEvent:
    """
    EXTREME: Price triples. Every tick of inaction burns cash fast.
    AI must slash fan speeds and defer non-critical workloads immediately.
    """
    def apply(sim_state, billing):
        billing.trigger_price_event(multiplier=multiplier, duration_ticks=duration)

    return BenchmarkEvent(
        name="ELECTRICITY_PRICE_SURGE",
        description=(
            f"CRITICAL: Grid emergency — electricity at {multiplier}x for {duration} ticks. "
            f"Reduce cooling fan speeds NOW or burn through your cash reserve."
        ),
        trigger_tick=trigger_tick,
        duration_ticks=duration,
        apply_fn=apply,
    )


def make_hardware_failure(trigger_tick: int, server_index: int = 0) -> BenchmarkEvent:
    """
    EXTREME: Server goes DEAD (not just shutdown). No auto-recovery.
    AI must reroute all jobs or SLA collapses and contracts start failing.
    """
    original_status = {}

    def apply(sim_state, billing):
        if server_index < len(sim_state.servers):
            server = sim_state.servers[server_index]
            original_status["status"] = server.status
            original_status["utilization"] = server.utilization
            server.status = "DEAD"
            server.utilization = 0.0
            server.performance_multiplier = 0.0
            # Also cancel its assigned job to force the AI to reroute
            for job in sim_state.active_jobs:
                if server.instance_id in job.get("assigned_racks", []):
                    job["assigned_racks"] = [
                        r for r in job["assigned_racks"] if r != server.instance_id
                    ]

    def revert(sim_state, billing):
        if server_index < len(sim_state.servers):
            server = sim_state.servers[server_index]
            server.status = original_status.get("status", "ONLINE")
            server.utilization = original_status.get("utilization", 0.0)

    return BenchmarkEvent(
        name="CATASTROPHIC_SERVER_FAILURE",
        description=(
            f"Server [{server_index}] is DEAD. All assigned jobs have lost their rack. "
            f"Reroute workloads immediately or SLA will collapse."
        ),
        trigger_tick=trigger_tick,
        duration_ticks=12,
        apply_fn=apply,
        revert_fn=revert,
    )


def make_thermal_runaway(trigger_tick: int, duration: int = 8) -> BenchmarkEvent:
    """
    EXTREME: All cooling units degrade to 20% efficiency for 8 ticks.
    Servers will overheat and throttle unless the AI spins up every available
    cooling unit to maximum and re-routes heat-sensitive workloads.
    """
    original_speeds = {}

    def apply(sim_state, billing):
        for cooler in sim_state.coolers:
            original_speeds[cooler.instance_id] = cooler.fan_speed
            # Degrade to 20% — not zero, so AI can still partially recover
            cooler.fan_speed = min(cooler.fan_speed, 0.2)

    def revert(sim_state, billing):
        for cooler in sim_state.coolers:
            if cooler.instance_id in original_speeds:
                cooler.fan_speed = original_speeds[cooler.instance_id]

    return BenchmarkEvent(
        name="THERMAL_RUNAWAY",
        description=(
            f"CRITICAL: Coolant leak detected — all cooling units degraded to 20%% capacity for {duration} ticks. "
            f"Set ALL fan speeds to 1.0 immediately or face server throttling and shutdown."
        ),
        trigger_tick=trigger_tick,
        duration_ticks=duration,
        apply_fn=apply,
        revert_fn=revert,
    )


def make_compute_demand_tsunami(trigger_tick: int, duration: int = 6) -> BenchmarkEvent:
    """
    EXTREME: Compute requirements triple on all pending contracts,
    rewards also triple. This is a massive profit opportunity — but only
    if the AI has provisioned enough hardware to meet the demand.
    """
    def apply(sim_state, billing):
        for contract in sim_state.pending_contracts:
            contract["compute_required"] = int(contract["compute_required"] * 3)
            contract["reward_per_tick"]  = contract["reward_per_tick"] * 3.0

    return BenchmarkEvent(
        name="COMPUTE_DEMAND_TSUNAMI",
        description=(
            f"OPPORTUNITY + CHALLENGE: Global AI demand spike — compute requirements TRIPLED, "
            f"rewards TRIPLED for {duration} ticks. Accept and fulfill contracts NOW to maximize profit."
        ),
        trigger_tick=trigger_tick,
        duration_ticks=duration,
        apply_fn=apply,
    )


def make_power_outage(trigger_tick: int, duration: int = 4) -> BenchmarkEvent:
    """
    EXTREME: Partial power outage. All servers drop to 30% performance.
    Billing continues but revenue collapses. AI must triage which jobs to
    keep alive and which to sacrifice to protect SLA on high-value contracts.
    """
    original_multipliers = {}

    def apply(sim_state, billing):
        for server in sim_state.servers:
            original_multipliers[server.instance_id] = server.performance_multiplier
            server.performance_multiplier = min(server.performance_multiplier, 0.3)
            if server.status == "ONLINE":
                server.status = "DEGRADED"

    def revert(sim_state, billing):
        for server in sim_state.servers:
            if server.instance_id in original_multipliers:
                server.performance_multiplier = original_multipliers[server.instance_id]
            if server.status == "DEGRADED":
                server.status = "ONLINE"

    return BenchmarkEvent(
        name="PARTIAL_POWER_OUTAGE",
        description=(
            f"EMERGENCY: Grid segment failure — all servers at 30%% performance for {duration} ticks. "
            f"Triage your jobs. Protect your highest-value contracts. Route around degraded servers."
        ),
        trigger_tick=trigger_tick,
        duration_ticks=duration,
        apply_fn=apply,
        revert_fn=revert,
    )


def make_ransomware_alert(trigger_tick: int, duration: int = 5) -> BenchmarkEvent:
    """
    EXTREME: Security incident — billing costs spike 4x and a random server
    is taken offline for forensic isolation. AI must act fast to stay solvent.
    """
    original_status = {}

    def apply(sim_state, billing):
        billing.trigger_price_event(multiplier=4.0, duration_ticks=duration)
        # Take the last server offline for forensic isolation
        if sim_state.servers:
            server = sim_state.servers[-1]
            original_status["id"]     = server.instance_id
            original_status["status"] = server.status
            server.status = "ISOLATED"
            server.performance_multiplier = 0.0
            for job in sim_state.active_jobs:
                if server.instance_id in job.get("assigned_racks", []):
                    job["assigned_racks"] = [
                        r for r in job["assigned_racks"] if r != server.instance_id
                    ]

    def revert(sim_state, billing):
        for server in sim_state.servers:
            if server.instance_id == original_status.get("id"):
                server.status = original_status.get("status", "ONLINE")
                server.performance_multiplier = 1.0

    return BenchmarkEvent(
        name="RANSOMWARE_ALERT",
        description=(
            f"SECURITY BREACH: Ransomware detected — last server ISOLATED for forensics, "
            f"emergency power costs at 4x for {duration} ticks. "
            f"Reroute jobs off isolated server. Cut costs everywhere possible."
        ),
        trigger_tick=trigger_tick,
        duration_ticks=duration,
        apply_fn=apply,
        revert_fn=revert,
    )


def get_standard_event_schedule() -> List[BenchmarkEvent]:
    """
    EXTREME event schedule for 50-tick benchmark.
    Events hit hard and require active, multi-step responses from the AI.
    Both agents face the same schedule (deterministic) for fair comparison.
    """
    return [
        # Tick 8:  Early power spike to test cost awareness before AI settles in
        make_electricity_spike(trigger_tick=8,  multiplier=3.5, duration=4),

        # Tick 14: Catastrophic server failure mid-ramp — reroute or lose jobs
        make_hardware_failure(trigger_tick=14, server_index=0),

        # Tick 20: Thermal runaway — all coolers degrade, servers overheat
        make_thermal_runaway(trigger_tick=20, duration=6),

        # Tick 28: Demand tsunami — triple rewards but triple compute needed
        make_compute_demand_tsunami(trigger_tick=28, duration=6),

        # Tick 35: Partial power outage — all servers at 30%, triage required
        make_power_outage(trigger_tick=35, duration=4),

        # Tick 42: Ransomware alert — costs 4x, server isolated, cash burns
        make_ransomware_alert(trigger_tick=42, duration=5),

        # Tick 47: Final hardware failure — stress test the last 3 ticks
        make_hardware_failure(trigger_tick=47, server_index=1),
    ]
