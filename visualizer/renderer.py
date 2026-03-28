import sys
import math
from typing import List, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from engine.grid import FacilityGrid, AMBIENT_TEMP, SAFE_TEMP, CRITICAL_TEMP
from engine.hardware import ServerRack, CoolingUnit


TILE_SIZE = 32
PANEL_WIDTH = 320
FPS = 10

COLOR_BACKGROUND   = (15,  15,  20)
COLOR_GRID_LINE    = (30,  30,  40)
COLOR_PANEL_BG     = (20,  20,  28)
COLOR_TEXT         = (220, 220, 220)
COLOR_TEXT_DIM     = (120, 120, 140)
COLOR_OK           = (60,  200, 100)
COLOR_WARN         = (240, 180,  40)
COLOR_CRIT         = (220,  50,  50)
COLOR_SERVER       = (60,  120, 200)
COLOR_COOLER       = (40,  180, 200)
COLOR_ARROW        = (255, 255, 255)

FACING_ARROWS = {
    "NORTH": (0, -1),
    "SOUTH": (0,  1),
    "EAST":  (1,  0),
    "WEST":  (-1, 0),
}


def _temp_to_color(temp: float) -> tuple:
    """
    Maps a temperature value to an RGB color.
    AMBIENT_TEMP (22.5C) -> deep blue
    SAFE_TEMP    (25.0C) -> cyan
    CRITICAL_TEMP(40.0C) -> bright red
    """
    t = max(0.0, min(1.0, (temp - AMBIENT_TEMP) / (CRITICAL_TEMP - AMBIENT_TEMP)))
    if t < 0.5:
        r = int(20  + t * 2 * 40)
        g = int(60  + t * 2 * 100)
        b = int(180 - t * 2 * 80)
    else:
        s = (t - 0.5) * 2
        r = int(60  + s * 160)
        g = int(160 - s * 130)
        b = int(100 - s * 90)
    return (r, g, b)


def _draw_arrow(surface, color, cx, cy, direction, size=8):
    dx, dy = direction
    tip   = (cx + dx * size,       cy + dy * size)
    left  = (cx - dy * size // 2,  cy + dx * size // 2)
    right = (cx + dy * size // 2,  cy - dx * size // 2)
    pygame.draw.polygon(surface, color, [tip, left, right])


class DataCenterRenderer:
    """
    Pygame-based 2D renderer for the ai-datacenter-sim.

    Left panel: the 2D grid with temperature heat map and hardware sprites.
    Right panel: real-time telemetry including tick, PUE, balance, and SLA.

    Usage:
        renderer = DataCenterRenderer(grid_width=20, grid_height=20)
        renderer.init()
        while running:
            renderer.render(grid, servers, coolers, metrics)
            running = renderer.handle_events()
        renderer.quit()
    """

    def __init__(self, grid_width: int = 20, grid_height: int = 20):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is not installed. Run: pip install pygame")
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.screen_w    = grid_width  * TILE_SIZE + PANEL_WIDTH
        self.screen_h    = grid_height * TILE_SIZE
        self.screen      = None
        self.clock       = None
        self.font_sm     = None
        self.font_md     = None
        self.font_lg     = None

    def init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("ai-datacenter-sim | Live Visualization")
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self.font_md = pygame.font.SysFont("monospace", 13)
        self.font_lg = pygame.font.SysFont("monospace", 16, bold=True)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(
        self,
        grid: FacilityGrid,
        servers: List[ServerRack],
        coolers: List[CoolingUnit],
        metrics: dict,
    ):
        self.screen.fill(COLOR_BACKGROUND)
        self._draw_grid(grid)
        self._draw_equipment(servers, coolers)
        self._draw_panel(metrics, servers, coolers)
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_grid(self, grid: FacilityGrid):
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                temp  = grid.get_tile_temp(x, y)
                color = _temp_to_color(temp)
                rect  = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLOR_GRID_LINE, rect, 1)

    def _draw_equipment(self, servers: List[ServerRack], coolers: List[CoolingUnit]):
        for server in servers:
            rx = server.x * TILE_SIZE
            ry = server.y * TILE_SIZE
            rect = pygame.Rect(rx + 2, ry + 2, TILE_SIZE - 4, TILE_SIZE - 4)

            if server.status == "SHUTDOWN":
                color = COLOR_CRIT
            elif server.status == "THROTTLING":
                color = COLOR_WARN
            else:
                color = COLOR_SERVER

            pygame.draw.rect(self.screen, color, rect, border_radius=3)

            cx = rx + TILE_SIZE // 2
            cy = ry + TILE_SIZE // 2
            _draw_arrow(self.screen, COLOR_ARROW, cx, cy, FACING_ARROWS[server.facing], size=7)

            label = self.font_sm.render(server.hardware_id.split("_")[-1][:3], True, (0, 0, 0))
            self.screen.blit(label, (rx + 3, ry + 3))

        for cooler in coolers:
            cx_px = cooler.x * TILE_SIZE
            cy_px = cooler.y * TILE_SIZE
            rect  = pygame.Rect(cx_px + 2, cy_px + 2, TILE_SIZE - 4, TILE_SIZE - 4)
            pygame.draw.rect(self.screen, COLOR_COOLER, rect, border_radius=3)
            ccx = cx_px + TILE_SIZE // 2
            ccy = cy_px + TILE_SIZE // 2
            _draw_arrow(self.screen, COLOR_ARROW, ccx, ccy, FACING_ARROWS[cooler.facing], size=7)

    def _draw_panel(self, metrics: dict, servers: List[ServerRack], coolers: List[CoolingUnit]):
        panel_x = self.grid_width * TILE_SIZE
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, self.screen_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)

        y = 14
        line_h = 22

        def draw_label(text, color=COLOR_TEXT_DIM, font=None):
            nonlocal y
            f = font or self.font_sm
            surf = f.render(text, True, color)
            self.screen.blit(surf, (panel_x + 12, y))
            y += line_h

        def draw_value(label, value, color=COLOR_TEXT):
            nonlocal y
            label_surf = self.font_sm.render(label, True, COLOR_TEXT_DIM)
            value_surf = self.font_md.render(str(value), True, color)
            self.screen.blit(label_surf, (panel_x + 12, y))
            self.screen.blit(value_surf, (panel_x + 150, y))
            y += line_h

        def divider():
            nonlocal y
            pygame.draw.line(
                self.screen, COLOR_GRID_LINE,
                (panel_x + 8, y), (panel_x + PANEL_WIDTH - 8, y)
            )
            y += 10

        draw_label("ai-datacenter-sim", COLOR_TEXT, self.font_lg)
        divider()

        tick   = metrics.get("tick", 0)
        pue    = metrics.get("pue", 0.0)
        bal    = metrics.get("bank_balance", 0.0)
        sla    = metrics.get("sla_uptime_percent", 100.0)
        model  = metrics.get("model_name", "unknown")
        rev    = metrics.get("revenue_this_tick", 0.0)
        elec   = metrics.get("electricity_cost_this_tick", 0.0)

        draw_label(f"Model: {model[:26]}", COLOR_TEXT)
        draw_value("Tick",          f"{tick}")
        divider()

        pue_color = COLOR_OK if pue < 1.3 else COLOR_WARN if pue < 1.6 else COLOR_CRIT
        draw_value("PUE",           f"{pue:.4f}",  pue_color)
        sla_color = COLOR_OK if sla >= 95 else COLOR_WARN if sla >= 80 else COLOR_CRIT
        draw_value("SLA Uptime",    f"{sla:.1f}%", sla_color)
        draw_value("Balance",       f"${bal:,.0f}")
        draw_value("Revenue/tick",  f"${rev:,.0f}")
        draw_value("Power cost/tick",f"${elec:,.0f}")
        divider()

        draw_label("HARDWARE", COLOR_TEXT_DIM)
        draw_value("Servers",  str(len(servers)))
        draw_value("Coolers",  str(len(coolers)))

        online   = sum(1 for s in servers if s.status == "ONLINE")
        throttle = sum(1 for s in servers if s.status == "THROTTLING")
        shutdown = sum(1 for s in servers if s.status == "SHUTDOWN")
        draw_value("Online",   str(online),   COLOR_OK)
        draw_value("Throttling",str(throttle), COLOR_WARN)
        draw_value("Shutdown", str(shutdown),  COLOR_CRIT)
        divider()

        draw_label("LEGEND", COLOR_TEXT_DIM)
        draw_label("Blue tile  = cold (22C)",  (100, 140, 220))
        draw_label("Red tile   = critical (40C)", (220, 80, 80))
        draw_label("Blue rect  = server",       COLOR_SERVER)
        draw_label("Cyan rect  = cooler",       COLOR_COOLER)
        draw_label("Arrow      = facing/intake", COLOR_ARROW)
        draw_label("ESC to quit", COLOR_TEXT_DIM)

    def quit(self):
        pygame.quit()
