from dataclasses import dataclass, field
from typing import Optional


DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

DIRECTION_VECTORS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}

OPPOSITE_DIRECTION = {
    "NORTH": "SOUTH",
    "SOUTH": "NORTH",
    "EAST": "WEST",
    "WEST": "EAST",
}


@dataclass
class ServerRack:
    """
    Represents a single server rack tile on the facility grid.

    Directional airflow:
        - Intake: the tile in the direction the server is FACING.
        - Exhaust: the tile directly BEHIND the server (opposite of facing).

    Performance is degraded by thermal throttling when intake air is too hot.
    """

    hardware_id: str
    instance_id: str
    x: int
    y: int
    facing: str
    cost: float
    compute_units: int
    power_idle_kw: float
    power_max_kw: float
    safe_temp_c: float = 25.0
    critical_temp_c: float = 40.0

    utilization: float = 0.0
    intake_temp_c: float = 22.5
    exhaust_temp_c: float = 22.5
    performance_multiplier: float = 1.0
    status: str = "ONLINE"
    assigned_job_id: Optional[str] = None

    @property
    def current_power_kw(self) -> float:
        return self.power_idle_kw + (self.power_max_kw - self.power_idle_kw) * self.utilization

    @property
    def heat_generated_kw(self) -> float:
        return self.current_power_kw

    @property
    def intake_vector(self) -> tuple:
        dx, dy = DIRECTION_VECTORS[self.facing]
        return (self.x + dx, self.y + dy)

    @property
    def exhaust_vector(self) -> tuple:
        opposite = OPPOSITE_DIRECTION[self.facing]
        dx, dy = DIRECTION_VECTORS[opposite]
        return (self.x + dx, self.y + dy)

    def update_throttle(self):
        """
        Recalculates the performance multiplier based on current intake temperature.
        Linearly degrades performance between safe_temp_c and critical_temp_c.
        Shuts down if critical temperature is exceeded.
        """
        if self.intake_temp_c <= self.safe_temp_c:
            self.performance_multiplier = 1.0
            self.status = "ONLINE"
        elif self.intake_temp_c >= self.critical_temp_c:
            self.performance_multiplier = 0.0
            self.status = "SHUTDOWN"
        else:
            ratio = (self.intake_temp_c - self.safe_temp_c) / (self.critical_temp_c - self.safe_temp_c)
            self.performance_multiplier = round(1.0 - ratio, 4)
            self.status = "THROTTLING"

    def effective_compute(self) -> float:
        return self.compute_units * self.utilization * self.performance_multiplier

    def to_dict(self) -> dict:
        return {
            "id": self.instance_id,
            "type": self.hardware_id,
            "position": {"x": self.x, "y": self.y},
            "facing": self.facing,
            "status": self.status,
            "utilization": round(self.utilization, 3),
            "intake_temp_c": round(self.intake_temp_c, 2),
            "exhaust_temp_c": round(self.exhaust_temp_c, 2),
            "performance_multiplier": round(self.performance_multiplier, 4),
            "current_power_kw": round(self.current_power_kw, 3),
            "assigned_job_id": self.assigned_job_id,
        }


@dataclass
class CoolingUnit:
    """
    Represents a single cooling unit tile on the facility grid.

    The cooling unit removes heat from the room by pushing cold air
    forward in its facing direction across its airflow range.

    Higher fan_speed removes more heat but consumes more electricity,
    directly degrading PUE.
    """

    hardware_id: str
    instance_id: str
    x: int
    y: int
    facing: str
    cost: float
    cooling_capacity_kw: float
    power_max_kw: float
    airflow_range: int

    fan_speed: float = 0.8
    status: str = "ONLINE"

    @property
    def current_power_kw(self) -> float:
        return self.power_max_kw * self.fan_speed

    @property
    def active_cooling_kw(self) -> float:
        return self.cooling_capacity_kw * self.fan_speed

    def get_airflow_tiles(self) -> list:
        """
        Returns the list of (x, y) tile coordinates that this cooling unit
        pushes cold air into, based on facing direction and airflow range.
        """
        dx, dy = DIRECTION_VECTORS[self.facing]
        tiles = []
        for i in range(1, self.airflow_range + 1):
            tiles.append((self.x + dx * i, self.y + dy * i))
        return tiles

    def to_dict(self) -> dict:
        return {
            "id": self.instance_id,
            "type": self.hardware_id,
            "position": {"x": self.x, "y": self.y},
            "facing": self.facing,
            "status": self.status,
            "fan_speed": round(self.fan_speed, 2),
            "active_cooling_kw": round(self.active_cooling_kw, 3),
            "current_power_kw": round(self.current_power_kw, 3),
        }
