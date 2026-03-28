from typing import List
from engine.grid import FacilityGrid, AMBIENT_TEMP, SAFE_TEMP, CRITICAL_TEMP
from engine.hardware import ServerRack, CoolingUnit


HEAT_TRANSFER_COEFFICIENT = 1.0
COOLING_TRANSFER_COEFFICIENT = 0.8


def apply_server_airflow(grid: FacilityGrid, servers: List[ServerRack]):
    """
    Applies directional airflow for all server racks.

    Step 1: Read intake temperature from the tile in front of the server.
    Step 2: Calculate exhaust temperature as intake + heat generated.
    Step 3: Dump exhaust heat into the tile behind the server.
    Step 4: Update the server's throttling state based on its intake temp.
    """
    for server in servers:
        if server.status == "SHUTDOWN" and server.utilization == 0.0:
            continue

        intake_x, intake_y = server.intake_vector
        exhaust_x, exhaust_y = server.exhaust_vector

        intake_temp = grid.get_tile_temp(intake_x, intake_y)
        server.intake_temp_c = intake_temp

        exhaust_heat = server.heat_generated_kw * HEAT_TRANSFER_COEFFICIENT
        server.exhaust_temp_c = round(intake_temp + exhaust_heat, 2)

        grid.apply_heat_delta(exhaust_x, exhaust_y, exhaust_heat)

        server.update_throttle()


def apply_cooling_airflow(grid: FacilityGrid, coolers: List[CoolingUnit]):
    """
    Applies directional cooling airflow for all cooling units.

    Cold air is pushed into each tile along the cooling unit's airflow path.
    The total cooling capacity is distributed evenly across all airflow tiles.
    """
    for cooler in coolers:
        if cooler.status != "ONLINE":
            continue

        airflow_tiles = cooler.get_airflow_tiles()
        if not airflow_tiles:
            continue

        cooling_per_tile = (
            cooler.active_cooling_kw * COOLING_TRANSFER_COEFFICIENT / len(airflow_tiles)
        )

        for tile_x, tile_y in airflow_tiles:
            if grid.is_valid_position(tile_x, tile_y):
                current_temp = grid.get_tile_temp(tile_x, tile_y)
                new_temp = max(AMBIENT_TEMP, current_temp - cooling_per_tile)
                delta = new_temp - current_temp
                grid.apply_heat_delta(tile_x, tile_y, delta)


def calculate_pue(servers: List[ServerRack], coolers: List[CoolingUnit]) -> float:
    """
    Calculates Power Usage Effectiveness.
    PUE = Total Facility Power / IT Equipment Power.
    A perfect PUE is 1.0. Any overhead from cooling increases this value.
    """
    it_power = sum(s.current_power_kw for s in servers)
    cooling_power = sum(c.current_power_kw for c in coolers)
    total_power = it_power + cooling_power

    if it_power == 0:
        return 0.0

    return round(total_power / it_power, 4)


def calculate_total_power_kw(
    servers: List[ServerRack], coolers: List[CoolingUnit]
) -> dict:
    """
    Returns a breakdown of total power consumption for the current tick.
    """
    it_power = round(sum(s.current_power_kw for s in servers), 3)
    cooling_power = round(sum(c.current_power_kw for c in coolers), 3)
    return {
        "it_power_kw": it_power,
        "cooling_power_kw": cooling_power,
        "total_power_kw": round(it_power + cooling_power, 3),
    }


def run_physics_tick(
    grid: FacilityGrid,
    servers: List[ServerRack],
    coolers: List[CoolingUnit],
):
    """
    Master physics tick function. Called once per simulation step.
    Order of operations matters:
        1. Apply server exhaust heat to the grid.
        2. Apply cooling unit cold airflow to the grid.
        3. Diffuse ambient heat across empty tiles.
    This ordering ensures that heat sources and sinks are resolved
    before natural diffusion spreads the residual heat.
    """
    apply_server_airflow(grid, servers)
    apply_cooling_airflow(grid, coolers)
    grid.diffuse_heat()
