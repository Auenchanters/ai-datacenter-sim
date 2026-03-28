import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


AMBIENT_TEMP = 22.5
SAFE_TEMP = 25.0
CRITICAL_TEMP = 40.0
DIFFUSION_ALPHA = 0.15


class FacilityGrid:
    """
    Represents the physical 2D floor plan of the data center.
    Each tile holds a temperature value in Celsius.
    The grid updates its thermal state every simulation tick.
    """

    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.temperature: np.ndarray = np.full(
            (height, width), fill_value=AMBIENT_TEMP, dtype=np.float64
        )
        self.occupied: np.ndarray = np.zeros((height, width), dtype=bool)

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_tile_free(self, x: int, y: int) -> bool:
        return self.is_valid_position(x, y) and not self.occupied[y][x]

    def place_equipment(self, x: int, y: int):
        if not self.is_valid_position(x, y):
            raise ValueError(f"Position ({x}, {y}) is out of grid bounds.")
        if self.occupied[y][x]:
            raise ValueError(f"Tile ({x}, {y}) is already occupied.")
        self.occupied[y][x] = True

    def remove_equipment(self, x: int, y: int):
        if self.is_valid_position(x, y):
            self.occupied[y][x] = False

    def apply_heat_delta(self, x: int, y: int, delta: float):
        if self.is_valid_position(x, y):
            self.temperature[y][x] += delta

    def diffuse_heat(self):
        """
        Applies Laplacian heat diffusion across the entire grid.
        Heat spreads from hotter tiles to cooler adjacent tiles.
        Formula: T(x,y)_t+1 = T(x,y)_t + alpha * (sum of 4 neighbours - 4 * T(x,y)_t)
        """
        padded = np.pad(self.temperature, pad_width=1, mode="edge")
        laplacian = (
            padded[:-2, 1:-1]  # North neighbour
            + padded[2:, 1:-1]  # South neighbour
            + padded[1:-1, :-2]  # West neighbour
            + padded[1:-1, 2:]  # East neighbour
            - 4 * self.temperature
        )
        self.temperature += DIFFUSION_ALPHA * laplacian
        self.temperature = np.clip(self.temperature, AMBIENT_TEMP, 200.0)

    def get_hotspots(self, threshold: float = SAFE_TEMP) -> List[dict]:
        """
        Returns a list of tiles where temperature exceeds the threshold.
        Used to build the token-efficient state payload for the AI.
        """
        hotspots = []
        ys, xs = np.where(self.temperature > threshold)
        for y, x in zip(ys, xs):
            temp = round(float(self.temperature[y][x]), 2)
            warning = "CRITICAL_HEAT_DETECTED" if temp >= CRITICAL_TEMP else "ELEVATED_TEMP"
            hotspots.append({"x": int(x), "y": int(y), "temp_c": temp, "warning": warning})
        return hotspots

    def get_tile_temp(self, x: int, y: int) -> float:
        if self.is_valid_position(x, y):
            return round(float(self.temperature[y][x]), 2)
        return AMBIENT_TEMP

    def reset(self):
        self.temperature = np.full(
            (self.height, self.width), fill_value=AMBIENT_TEMP, dtype=np.float64
        )
        self.occupied = np.zeros((self.height, self.width), dtype=bool)
