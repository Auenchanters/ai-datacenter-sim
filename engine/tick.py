from typing import List, Optional
from engine.grid import FacilityGrid
from engine.hardware import ServerRack, CoolingUnit
from engine.physics import run_physics_tick, calculate_pue, calculate_total_power_kw


class SimulationState:
    """
    Holds the complete mutable state of the simulation.
    This object is passed into and mutated by each tick.
    """

    def __init__(
        self,
        grid: FacilityGrid,
        starting_budget: float = 500_000.0,
        electricity_cost_per_kwh: float = 0.12,
        tick_duration_hours: float = 1.0,
    ):
        self.grid = grid
        self.bank_balance: float = starting_budget
        self.electricity_cost_per_kwh: float = electricity_cost_per_kwh
        self.tick_duration_hours: float = tick_duration_hours

        self.servers: List[ServerRack] = []
        self.coolers: List[CoolingUnit] = []
        self.active_jobs: list = []
        self.pending_contracts: list = []
        self.completed_jobs: list = []
        self.failed_jobs: list = []

        self.current_tick: int = 0
        self.current_pue: float = 0.0
        self.current_power_breakdown: dict = {}
        self.total_revenue_earned: float = 0.0
        self.total_electricity_spent: float = 0.0
        self.sla_uptime_percent: float = 100.0

        self._total_job_ticks: int = 0
        self._successful_job_ticks: int = 0

    def get_equipment_by_id(self, instance_id: str):
        for s in self.servers:
            if s.instance_id == instance_id:
                return s
        for c in self.coolers:
            if c.instance_id == instance_id:
                return c
        return None

    def get_server_by_id(self, instance_id: str) -> Optional[ServerRack]:
        for s in self.servers:
            if s.instance_id == instance_id:
                return s
        return None

    def get_cooler_by_id(self, instance_id: str) -> Optional[CoolingUnit]:
        for c in self.coolers:
            if c.instance_id == instance_id:
                return c
        return None


def _process_workloads(state: SimulationState) -> float:
    """
    Iterates all active jobs. For each job:
    - Calculates effective compute delivered based on server performance multipliers.
    - Pays the AI agent the contract reward if the job is being served.
    - Marks jobs as failed if assigned servers are in SHUTDOWN status.
    - Decrements job expiry counters and removes expired jobs.
    Returns total revenue earned this tick.
    """
    revenue = 0.0
    jobs_to_remove = []

    for job in state.active_jobs:
        assigned_servers = [
            state.get_server_by_id(rack_id)
            for rack_id in job.get("assigned_racks", [])
            if state.get_server_by_id(rack_id) is not None
        ]

        if not assigned_servers:
            job["ticks_remaining"] -= 1
            if job["ticks_remaining"] <= 0:
                state.failed_jobs.append(job)
                jobs_to_remove.append(job)
            continue

        all_shutdown = all(s.status == "SHUTDOWN" for s in assigned_servers)
        state._total_job_ticks += 1

        if all_shutdown:
            job["ticks_remaining"] -= 1
            state.failed_jobs.append(dict(job))
            jobs_to_remove.append(job)
        else:
            avg_multiplier = sum(s.performance_multiplier for s in assigned_servers) / len(assigned_servers)
            tick_revenue = job["reward_per_tick"] * avg_multiplier
            revenue += tick_revenue
            state._successful_job_ticks += 1
            job["ticks_remaining"] -= 1
            if job["ticks_remaining"] <= 0:
                state.completed_jobs.append(job)
                jobs_to_remove.append(job)

    for job in jobs_to_remove:
        if job in state.active_jobs:
            state.active_jobs.remove(job)

    return round(revenue, 4)


def _calculate_electricity_cost(state: SimulationState) -> float:
    """
    Calculates the electricity bill for this tick.
    Cost = Total Power (kW) * Tick Duration (hours) * Cost per kWh.
    """
    total_kw = state.current_power_breakdown.get("total_power_kw", 0.0)
    cost = total_kw * state.tick_duration_hours * state.electricity_cost_per_kwh
    return round(cost, 4)


def _update_sla(state: SimulationState):
    """
    Recalculates the running SLA uptime percentage.
    SLA = (successful job ticks / total job ticks) * 100.
    """
    if state._total_job_ticks == 0:
        state.sla_uptime_percent = 100.0
    else:
        state.sla_uptime_percent = round(
            (state._successful_job_ticks / state._total_job_ticks) * 100, 2
        )


def _expire_pending_contracts(state: SimulationState):
    """
    Decrements expiry timers on pending contracts.
    Removes contracts the AI failed to accept before they expired.
    """
    expired = []
    for contract in state.pending_contracts:
        contract["expires_in_ticks"] -= 1
        if contract["expires_in_ticks"] <= 0:
            expired.append(contract)
    for c in expired:
        state.pending_contracts.remove(c)


def step(state: SimulationState):
    """
    Master tick function. Advances the simulation by one unit of time.

    Order of operations:
        1. Run physics: server heat, cooling airflow, heat diffusion.
        2. Calculate power breakdown and PUE.
        3. Process active workloads and collect revenue.
        4. Deduct electricity costs from bank balance.
        5. Expire pending contracts the AI has not accepted.
        6. Increment tick counter and update SLA uptime.
    """
    run_physics_tick(state.grid, state.servers, state.coolers)

    state.current_power_breakdown = calculate_total_power_kw(state.servers, state.coolers)
    state.current_pue = calculate_pue(state.servers, state.coolers)

    revenue = _process_workloads(state)
    state.total_revenue_earned += revenue
    state.bank_balance += revenue

    electricity_cost = _calculate_electricity_cost(state)
    state.total_electricity_spent += electricity_cost
    state.bank_balance -= electricity_cost
    state.bank_balance = round(state.bank_balance, 4)

    _expire_pending_contracts(state)
    _update_sla(state)

    state.current_tick += 1

    return {
        "tick": state.current_tick,
        "revenue_this_tick": revenue,
        "electricity_cost_this_tick": electricity_cost,
        "bank_balance": state.bank_balance,
        "pue": state.current_pue,
        "sla_uptime_percent": state.sla_uptime_percent,
        "power_breakdown": state.current_power_breakdown,
    }
