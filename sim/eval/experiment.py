from __future__ import annotations

from typing import Dict, List


def collect_metrics(
    frame_count: int, collision_count: int = 0, route_completion: float = 0.0
) -> Dict[str, float]:
    return {
        "frame_count": float(frame_count),
        "collision_count": float(collision_count),
        "route_completion": float(route_completion),
    }


def run_experiment() -> Dict[str, float]:
    """Minimal evaluation entrypoint for a CARLA experiment."""
    return collect_metrics(frame_count=100, collision_count=0, route_completion=1.0)
