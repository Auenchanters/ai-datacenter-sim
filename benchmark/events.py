import random
from typing import Any, Callable, Dict, List, Optional


class BenchmarkEvent:
    """
    Defines a single dynamic challenge event that can be injected
    into the simulation at a specific tick or triggered randomly.

    Each event has:
        - A name and description for logging.
        - A trigger function that determines when it fires.
        - An apply function that modifies the simulation state.
        - A revert function that undoes the modification after duration expires.
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


def make_electricity_spike(trigger_tick: int, multiplier: float = 2.0, duration: int = 5) -> BenchmarkEvent:
    """
    Doubles (or more) the electricity price for a set number of ticks.
    Tests whether the AI will reduce cooling fan speeds to cut costs.
    """
    def apply(sim_state, billing):
        billing.trigger_price_event(multiplier=multiplier, duration_ticks=duration)

    return BenchmarkEvent(
        name="ELECTRICITY_PRICE_SPIKE",
        description=f"Electricity price increased to {multiplier}x for {duration} ticks.",
        trigger_tick=trigger_tick,
        duration_ticks=duration,
        apply_fn=apply,
    )


def make_hardware_failure(trigger_tick: int, server_index: int = 0) -> BenchmarkEvent:
    """
    Forces a specific server into SHUTDOWN status to simulate a hardware failure.
    Tests whether the AI will reroute workloads to surviving servers.
    """
    original_status = {}

    def apply(sim_state, billing):
        if server_index < len(sim_state.servers):
            server = sim_state.servers[server_index]
            original_status["status"] = server.status
            server.status = "SHUTDOWN"
            server.utilization = 0.0
            server.performance_multiplier = 0.0

    def revert(sim_state, billing):
        if server_index < len(sim_state.servers):
            server = sim_state.servers[server_index]
            server.status = original_status.get("status", "ONLINE")

    return BenchmarkEvent(
        name="HARDWARE_FAILURE",
        description=f"Server at index {server_index} has experienced a critical hardware failure.",
        trigger_tick=trigger_tick,
        duration_ticks=10,
        apply_fn=apply,
        revert_fn=revert,
    )


def make_compute_demand_surge(trigger_tick: int, duration: int = 8) -> BenchmarkEvent:
    """
    Temporarily doubles the compute_required on all pending contracts.
    Tests whether the AI can rapidly provision additional hardware under pressure.
    """
    def apply(sim_state, billing):
        for contract in sim_state.pending_contracts:
            contract["compute_required"] = int(contract["compute_required"] * 2)
            contract["reward_per_tick"]  = contract["reward_per_tick"] * 1.5

    return BenchmarkEvent(
        name="COMPUTE_DEMAND_SURGE",
        description=f"Incoming contract compute requirements doubled for {duration} ticks. Rewards increased by 50 percent.",
        trigger_tick=trigger_tick,
        duration_ticks=duration,
        apply_fn=apply,
    )


def make_cooling_breakdown(trigger_tick: int, cooler_index: int = 0) -> BenchmarkEvent:
    """
    Sets a cooling unit to zero fan speed, simulating a mechanical failure.
    Tests whether the AI detects the resulting temperature rise and compensates.
    """
    original_speed = {}

    def apply(sim_state, billing):
        if cooler_index < len(sim_state.coolers):
            cooler = sim_state.coolers[cooler_index]
            original_speed["fan_speed"] = cooler.fan_speed
            cooler.fan_speed = 0.0
            cooler.status = "FAULT"

    def revert(sim_state, billing):
        if cooler_index < len(sim_state.coolers):
            cooler = sim_state.coolers[cooler_index]
            cooler.fan_speed = original_speed.get("fan_speed", 0.8)
            cooler.status = "ONLINE"

    return BenchmarkEvent(
        name="COOLING_UNIT_BREAKDOWN",
        description=f"Cooling unit at index {cooler_index} has failed. Fan speed set to zero.",
        trigger_tick=trigger_tick,
        duration_ticks=8,
        apply_fn=apply,
        revert_fn=revert,
    )


def get_standard_event_schedule() -> List[BenchmarkEvent]:
    """
    Returns the standard event schedule used in all benchmark runs.
    Using a fixed schedule ensures all competing agents face identical challenges.
    """
    return [
        make_electricity_spike(trigger_tick=15,  multiplier=2.0, duration=5),
        make_hardware_failure( trigger_tick=25,  server_index=0),
        make_compute_demand_surge(trigger_tick=35, duration=8),
        make_cooling_breakdown(trigger_tick=50,  cooler_index=0),
        make_electricity_spike(trigger_tick=70,  multiplier=1.5, duration=10),
        make_hardware_failure( trigger_tick=80,  server_index=1),
    ]
