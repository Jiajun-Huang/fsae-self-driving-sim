from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .base_detector import BaseDetector
from .cone_fusion import FusedCone, distance_2d


Point2 = tuple[float, float]


@dataclass(frozen=True)
class DetectedLane:
    left_boundary: list[Point2]
    right_boundary: list[Point2]
    centerline: list[Point2]
    confidence: float


class LaneDetector(BaseDetector):
    def __init__(
        self,
        expected_width_m: float = 6.0,
        max_segment_length_m: float = 12.0,
        max_search_points: int = 128,
    ) -> None:
        super().__init__("graph_search_lane_detector")
        self.expected_width_m = expected_width_m
        self.max_segment_length_m = max_segment_length_m
        self.max_search_points = max_search_points

    def process(self, data: Any) -> dict[str, Any]:
        cones = self._read_cones(data)
        lane = self.detect_lane(cones)
        if lane is None:
            return {
                "status": "no_lane",
                "lanes": [],
                "centerline": [],
            }
        return {
            "status": "ok",
            "lanes": [
                {
                    "left_boundary": lane.left_boundary,
                    "right_boundary": lane.right_boundary,
                    "confidence": lane.confidence,
                }
            ],
            "centerline": lane.centerline,
        }

    def detect_lane(self, cones: list[FusedCone]) -> DetectedLane | None:
        left = self._ordered_boundary(cones, preferred_color="blue")
        right = self._ordered_boundary(cones, preferred_color="yellow")
        if len(left) < 2 or len(right) < 2:
            return None

        left, right = self._align_boundary_pair(left, right)
        centerline = [((l[0] + r[0]) * 0.5, (l[1] + r[1]) * 0.5) for l, r in zip(left, right)]
        confidence = self._lane_confidence(left, right)
        return DetectedLane(
            left_boundary=left,
            right_boundary=right,
            centerline=centerline,
            confidence=confidence,
        )

    def _read_cones(self, data: Any) -> list[FusedCone]:
        if isinstance(data, dict):
            cones = data.get("cones", [])
        else:
            cones = data

        result: list[FusedCone] = []
        for item in cones:
            if isinstance(item, FusedCone):
                result.append(item)
                continue
            # This keeps the detector usable with simple dict logs from experiments.
            result.append(
                FusedCone(
                    color=str(item.get("color", "unknown")),
                    x_m=float(item.get("x_m", item.get("x", 0.0))),
                    y_m=float(item.get("y_m", item.get("y", 0.0))),
                    confidence=float(item.get("confidence", 1.0)),
                    source=str(item.get("source", "unknown")),
                )
            )
        return result

    def _ordered_boundary(self, cones: list[FusedCone], preferred_color: str) -> list[Point2]:
        colored = [cone for cone in cones if cone.color == preferred_color]
        neutral = [cone for cone in cones if cone.color == "unknown"]
        if len(colored) >= 6:
            return self._ordered_closed_boundary(colored)

        candidates = sorted(
            colored + neutral,
            key=lambda cone: (cone.x_m * cone.x_m + cone.y_m * cone.y_m, -cone.confidence),
        )[: self.max_search_points]
        if not candidates:
            return []

        start = min(candidates, key=lambda cone: cone.x_m * cone.x_m + cone.y_m * cone.y_m)
        path = [start]
        remaining = [cone for cone in candidates if cone is not start]

        while remaining:
            current = path[-1]
            next_cone = self._next_boundary_cone(path, remaining, preferred_color)
            if next_cone is None:
                break
            path.append(next_cone)
            remaining.remove(next_cone)

        return [(cone.x_m, cone.y_m) for cone in path]

    def _ordered_closed_boundary(self, cones: list[FusedCone]) -> list[Point2]:
        cx = sum(cone.x_m for cone in cones) / len(cones)
        cy = sum(cone.y_m for cone in cones) / len(cones)
        ordered = sorted(cones, key=lambda cone: math.atan2(cone.y_m - cy, cone.x_m - cx))
        start = min(range(len(ordered)), key=lambda index: ordered[index].x_m ** 2 + ordered[index].y_m ** 2)
        rotated = ordered[start:] + ordered[:start]
        return [(cone.x_m, cone.y_m) for cone in rotated[: self.max_search_points]]

    def _align_boundary_pair(self, left: list[Point2], right: list[Point2]) -> tuple[list[Point2], list[Point2]]:
        pair_count = min(len(left), len(right))
        left = left[:pair_count]
        right = right[:pair_count]
        nearest_left: list[Point2] = []
        nearest_right: list[Point2] = []
        unused_right = list(right)

        for left_point in left:
            if not unused_right:
                break
            # FSAE cone lanes are sparse; local nearest-width matching is more stable than index matching.
            right_point = min(
                unused_right,
                key=lambda candidate: abs(distance_2d(left_point, candidate) - self.expected_width_m),
            )
            width = distance_2d(left_point, right_point)
            if 0.5 * self.expected_width_m <= width <= 1.8 * self.expected_width_m:
                nearest_left.append(left_point)
                nearest_right.append(right_point)
                unused_right.remove(right_point)

        candidates = [right, list(reversed(right))]
        best_right = right
        best_error = float("inf")

        for candidate in candidates:
            for shift in range(pair_count):
                shifted = candidate[shift:] + candidate[:shift]
                # Closed FSAE tracks have no natural list origin, so choose the cyclic alignment with stable width.
                mean_error = sum(
                    abs(distance_2d(l, r) - self.expected_width_m)
                    for l, r in zip(left, shifted)
                ) / pair_count
                if mean_error < best_error:
                    best_error = mean_error
                    best_right = shifted

        cyclic_error = best_error
        if nearest_left:
            nearest_error = sum(
                abs(distance_2d(l, r) - self.expected_width_m)
                for l, r in zip(nearest_left, nearest_right)
            ) / len(nearest_left)
            if nearest_error <= cyclic_error:
                return nearest_left, nearest_right

        return left, best_right

    def _next_boundary_cone(
        self,
        path: list[FusedCone],
        remaining: list[FusedCone],
        preferred_color: str,
    ) -> FusedCone | None:
        current = path[-1]
        previous_heading = self._path_heading(path)
        scored: list[tuple[float, FusedCone]] = []

        for cone in remaining:
            step = distance_2d((current.x_m, current.y_m), (cone.x_m, cone.y_m))
            if step > self.max_segment_length_m:
                continue
            heading = math.atan2(cone.y_m - current.y_m, cone.x_m - current.x_m)
            turn_cost = abs(self._angle_delta(previous_heading, heading)) if previous_heading is not None else 0.0
            color_penalty = 0.0 if cone.color == preferred_color else 1.5
            # This is the graph-search ranking heuristic: prefer close, smooth, high-confidence extensions.
            score = step + 2.0 * turn_cost + color_penalty - 0.5 * cone.confidence
            scored.append((score, cone))

        return min(scored)[1] if scored else None

    def _path_heading(self, path: list[FusedCone]) -> float | None:
        if len(path) < 2:
            return None
        a = path[-2]
        b = path[-1]
        return math.atan2(b.y_m - a.y_m, b.x_m - a.x_m)

    def _lane_confidence(self, left: list[Point2], right: list[Point2]) -> float:
        widths = [distance_2d(l, r) for l, r in zip(left, right)]
        if not widths:
            return 0.0
        mean_width_error = sum(abs(width - self.expected_width_m) for width in widths) / len(widths)
        return max(0.0, min(1.0, 1.0 - mean_width_error / self.expected_width_m))

    def _angle_delta(self, a: float, b: float) -> float:
        delta = b - a
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta
