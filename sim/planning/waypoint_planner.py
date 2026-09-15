from __future__ import annotations

from typing import List, Tuple


class WaypointPlanner:
    """Minimal baseline planner for a straight path or waypoint following."""

    def __init__(self, target_speed: float = 8.0):
        self.target_speed = target_speed

    def generate_waypoints(
        self, start: Tuple[float, float], end: Tuple[float, float], steps: int = 10
    ) -> List[Tuple[float, float]]:
        x1, y1 = start
        x2, y2 = end
        xs = [x1 + (x2 - x1) * i / max(steps, 1) for i in range(steps + 1)]
        ys = [y1 + (y2 - y1) * i / max(steps, 1) for i in range(steps + 1)]
        return list(zip(xs, ys))

    def control_command(self, current_speed: float) -> float:
        # Simple proportional throttle-like command; replace with real control logic.
        return self.target_speed - current_speed
