import uuid
from typing import Any, Dict, List, Tuple
from engine.grid import FacilityGrid
from engine.hardware import ServerRack, CoolingUnit, DIRECTIONS
from economy.catalog import HARDWARE_CATALOG, get_item
from economy.billing import BillingEngine


VALID_COMMANDS = {
    "BUY_EQUIPMENT",
    "ADJUST_COOLING",
    "ACCEPT_CONTRACT",
    "ROUTE_WORKLOAD",
}


class ParseError(Exception):
    pass


def _validate_position(position: Any, grid: FacilityGrid) -> Tuple[int, int]:
    if not isinstance(position, dict):
        raise ParseError("position must be a JSON object with 'x' and 'y' keys.")
    x = position.get("x")
    y = position.get("y")
    if not isinstance(x, int) or not isinstance(y, int):
        raise ParseError(f"position x and y must be integers. Got x={x}, y={y}.")
    if not grid.is_valid_position(x, y):
        raise ParseError(f"Position ({x}, {y}) is outside the grid bounds ({grid.width}x{grid.height}).")
    if not grid.is_tile_free(x, y):
        raise ParseError(f"Tile ({x}, {y}) is already occupied.")
    return x, y


def _validate_facing(facing: Any) -> str:
    if facing not in DIRECTIONS:
        raise ParseError(f"Invalid facing direction '{facing}'. Must be one of {DIRECTIONS}.")
    return facing


def _execute_buy_equipment(
    action: Dict[str, Any],
    grid: FacilityGrid,
    servers: List[ServerRack],
    coolers: List[CoolingUnit],
    billing: BillingEngine,
    tick: int,
) -> str:
    hardware_id = action.get("type")
    if hardware_id not in HARDWARE_CATALOG:
        raise ParseError(f"Unknown hardware type '{hardware_id}'. Check the catalog.")

    item = get_item(hardware_id)
    cost = item["cost"]

    if not billing.can_afford(cost):
        raise ParseError(
            f"Insufficient funds to buy '{hardware_id}'. "
            f"Cost: ${cost:,}. Balance: ${billing.bank_balance:,.2f}."
        )

    x, y = _validate_position(action.get("position"), grid)
    facing = _validate_facing(action.get("facing"))

    instance_id = f"{hardware_id.lower()}_{uuid.uuid4().hex[:6]}"

    if item["category"] == "server":
        rack = ServerRack(
            hardware_id=hardware_id,
            instance_id=instance_id,
            x=x,
            y=y,
            facing=facing,
            cost=cost,
            compute_units=item["compute_units"],
            power_idle_kw=item["power_idle_kw"],
            power_max_kw=item["power_max_kw"],
            safe_temp_c=item["safe_temp_c"],
            critical_temp_c=item["critical_temp_c"],
        )
        grid.place_equipment(x, y)
        servers.append(rack)
        billing.deduct_hardware_purchase(cost, hardware_id, tick)
        return f"Placed {hardware_id} at ({x},{y}) facing {facing}. Instance ID: {instance_id}."

    elif item["category"] == "cooling":
        cooler = CoolingUnit(
            hardware_id=hardware_id,
            instance_id=instance_id,
            x=x,
            y=y,
            facing=facing,
            cost=cost,
            cooling_capacity_kw=item["cooling_capacity_kw"],
            power_max_kw=item["power_max_kw"],
            airflow_range=item["airflow_range"],
        )
        grid.place_equipment(x, y)
        coolers.append(cooler)
        billing.deduct_hardware_purchase(cost, hardware_id, tick)
        return f"Placed {hardware_id} at ({x},{y}) facing {facing}. Instance ID: {instance_id}."

    raise ParseError(f"Unknown hardware category for '{hardware_id}'.")


def _execute_adjust_cooling(
    action: Dict[str, Any],
    coolers: List[CoolingUnit],
) -> str:
    target_id = action.get("target_id")
    fan_speed = action.get("fan_speed")

    if not isinstance(fan_speed, (int, float)) or not (0.0 <= fan_speed <= 1.0):
        raise ParseError(f"fan_speed must be a float between 0.0 and 1.0. Got {fan_speed}.")

    for cooler in coolers:
        if cooler.instance_id == target_id:
            cooler.fan_speed = round(float(fan_speed), 3)
            return f"Adjusted fan speed of '{target_id}' to {fan_speed}."

    raise ParseError(f"No cooling unit found with instance ID '{target_id}'.")


def _execute_accept_contract(
    action: Dict[str, Any],
    pending_contracts: list,
    active_jobs: list,
) -> str:
    job_id = action.get("job_id")
    for contract in pending_contracts:
        if contract["id"] == job_id:
            pending_contracts.remove(contract)
            active_jobs.append(contract)
            return f"Contract '{job_id}' accepted and moved to active jobs."
    raise ParseError(f"No pending contract found with ID '{job_id}'.")


def _execute_route_workload(
    action: Dict[str, Any],
    active_jobs: list,
    servers: List[ServerRack],
) -> str:
    job_id = action.get("job_id")
    rack_ids = action.get("rack_ids", [])

    if not isinstance(rack_ids, list) or len(rack_ids) == 0:
        raise ParseError("rack_ids must be a non-empty list of server instance IDs.")

    server_map = {s.instance_id: s for s in servers}
    for rid in rack_ids:
        if rid not in server_map:
            raise ParseError(f"Server instance ID '{rid}' does not exist.")

    for job in active_jobs:
        if job["id"] == job_id:
            total_compute = sum(
                server_map[rid].compute_units for rid in rack_ids
            )
            if total_compute < job["compute_required"]:
                raise ParseError(
                    f"Insufficient compute for job '{job_id}'. "
                    f"Required: {job['compute_required']} CU. "
                    f"Assigned racks provide: {total_compute} CU."
                )
            job["assigned_racks"] = rack_ids
            for rid in rack_ids:
                server_map[rid].utilization = min(
                    1.0,
                    job["compute_required"] / max(server_map[rid].compute_units, 1),
                )
                server_map[rid].assigned_job_id = job_id
            return f"Routed job '{job_id}' to racks {rack_ids}."

    raise ParseError(f"No active job found with ID '{job_id}'.")


def parse_and_execute(
    action_payload: Dict[str, Any],
    grid: FacilityGrid,
    servers: List[ServerRack],
    coolers: List[CoolingUnit],
    billing: BillingEngine,
    active_jobs: list,
    pending_contracts: list,
    tick: int,
) -> Dict[str, Any]:
    """
    Master parser. Accepts the AI's full action JSON, iterates each command,
    validates it, and executes it against the simulation state.

    Returns a results dict containing per-action success/failure messages
    and the AI's recorded thought chain for logging and debugging.
    """
    thoughts = action_payload.get("thoughts", "")
    actions = action_payload.get("actions", [])

    results = []

    if not isinstance(actions, list):
        return {
            "thoughts": thoughts,
            "results": [{"command": "UNKNOWN", "status": "ERROR", "message": "'actions' must be a JSON array."}],
        }

    for action in actions:
        command = action.get("command", "UNKNOWN")
        if command not in VALID_COMMANDS:
            results.append({
                "command": command,
                "status": "ERROR",
                "message": f"Unknown command '{command}'. Valid commands: {list(VALID_COMMANDS)}.",
            })
            continue

        try:
            if command == "BUY_EQUIPMENT":
                msg = _execute_buy_equipment(action, grid, servers, coolers, billing, tick)
            elif command == "ADJUST_COOLING":
                msg = _execute_adjust_cooling(action, coolers)
            elif command == "ACCEPT_CONTRACT":
                msg = _execute_accept_contract(action, pending_contracts, active_jobs)
            elif command == "ROUTE_WORKLOAD":
                msg = _execute_route_workload(action, active_jobs, servers)
            else:
                msg = "Command registered but has no executor."

            results.append({"command": command, "status": "OK", "message": msg})

        except ParseError as e:
            results.append({"command": command, "status": "ERROR", "message": str(e)})

    return {"thoughts": thoughts, "results": results}
