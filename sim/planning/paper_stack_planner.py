from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from sim.roadgen.geometry import Point2, polyline_length, resample_polyline
from sim.roadgen.models import Cone
from sim.roadgen.rules import ConeColor, ConeRole


@dataclass(frozen=True)
class ConeLandmark:
    x: float
    y: float
    color: str
    confidence: float = 1.0

    @property
    def point(self) -> Point2:
        return (self.x, self.y)


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw_rad: float
    speed_mps: float
    steer_rad: float = 0.0


@dataclass(frozen=True)
class ControlCommand:
    steer: float
    throttle: float
    brake: float
    target_speed_mps: float


@dataclass(frozen=True)
class PlannedPath:
    landmarks: list[ConeLandmark]
    triangles: list[tuple[int, int, int]]
    cross_edges: list[tuple[int, int]]
    raw_centerline: list[Point2]
    trajectory: list[Point2]
    target_speed_mps: float


class GraphSlamConeMapper:
    """Small landmark mapper shaped like the paper's GraphSLAM stage.

    The real KIT21d system optimizes a graph of vehicle poses and cone landmarks
    with g2o. For this simulator baseline we accept already-global cone
    observations, merge repeated detections, and keep stable landmark estimates.
    """

    def __init__(self, association_radius_m: float = 0.8) -> None:
        self.association_radius_m = association_radius_m
        self._landmarks: list[ConeLandmark] = []

    @property
    def landmarks(self) -> list[ConeLandmark]:
        return list(self._landmarks)

    def reset(self) -> None:
        self._landmarks.clear()

    def update_global_cones(self, cones: Iterable[Cone | ConeLandmark | dict]) -> list[ConeLandmark]:
        for cone in cones:
            landmark = self._read_landmark(cone)
            index = self._nearest_same_color(landmark)
            if index is None:
                self._landmarks.append(landmark)
                continue

            old = self._landmarks[index]
            total_confidence = max(old.confidence + landmark.confidence, 1e-6)
            # Confidence-weighted averaging mimics repeated landmark constraints.
            self._landmarks[index] = ConeLandmark(
                x=(old.x * old.confidence + landmark.x * landmark.confidence) / total_confidence,
                y=(old.y * old.confidence + landmark.y * landmark.confidence) / total_confidence,
                color=old.color,
                confidence=min(total_confidence, 10.0),
            )

        return self.landmarks

    def _nearest_same_color(self, landmark: ConeLandmark) -> int | None:
        best_index: int | None = None
        best_distance = self.association_radius_m
        for index, existing in enumerate(self._landmarks):
            if existing.color != landmark.color:
                continue
            distance = math.hypot(existing.x - landmark.x, existing.y - landmark.y)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _read_landmark(self, cone: Cone | ConeLandmark | dict) -> ConeLandmark:
        if isinstance(cone, ConeLandmark):
            return cone
        if isinstance(cone, Cone):
            return ConeLandmark(x=cone.x, y=cone.y, color=cone.color.value)

        return ConeLandmark(
            x=float(cone.get("x", cone.get("x_m", 0.0))),
            y=float(cone.get("y", cone.get("y_m", 0.0))),
            color=str(cone.get("color", ConeColor.ORANGE.value)),
            confidence=float(cone.get("confidence", 1.0)),
        )


class DelaunayTriangulation:
    """Dependency-free Bowyer-Watson Delaunay triangulation for sparse cones."""

    def triangulate(self, points: Sequence[Point2]) -> list[tuple[int, int, int]]:
        if len(points) < 3:
            return []

        pts = np.asarray(points, dtype=float)
        super_points = self._super_triangle(pts)
        all_points = np.vstack([pts, super_points])
        super_indices = {len(pts), len(pts) + 1, len(pts) + 2}
        triangles: list[tuple[int, int, int]] = [(len(pts), len(pts) + 1, len(pts) + 2)]

        for point_index in range(len(pts)):
            bad_triangles = [
                triangle
                for triangle in triangles
                if self._inside_circumcircle(all_points[point_index], triangle, all_points)
            ]
            boundary = self._polygon_boundary(bad_triangles)
            triangles = [triangle for triangle in triangles if triangle not in bad_triangles]
            triangles.extend((edge[0], edge[1], point_index) for edge in boundary)

        return [
            triangle
            for triangle in triangles
            if not any(index in super_indices for index in triangle)
            and abs(self._signed_area(triangle, all_points)) > 1e-9
        ]

    def _super_triangle(self, pts: np.ndarray) -> np.ndarray:
        minimum = pts.min(axis=0)
        maximum = pts.max(axis=0)
        center = (minimum + maximum) * 0.5
        span = max(float(np.max(maximum - minimum)), 1.0)
        size = span * 20.0
        return np.asarray(
            [
                (center[0] - size, center[1] - size),
                (center[0], center[1] + size),
                (center[0] + size, center[1] - size),
            ],
            dtype=float,
        )

    def _inside_circumcircle(
        self,
        point: np.ndarray,
        triangle: tuple[int, int, int],
        all_points: np.ndarray,
    ) -> bool:
        a, b, c = (all_points[index] for index in triangle)
        ax, ay = a - point
        bx, by = b - point
        cx, cy = c - point
        determinant = (
            (ax * ax + ay * ay) * (bx * cy - by * cx)
            - (bx * bx + by * by) * (ax * cy - ay * cx)
            + (cx * cx + cy * cy) * (ax * by - ay * bx)
        )
        orientation = self._signed_area(triangle, all_points)
        return determinant > 1e-9 if orientation > 0.0 else determinant < -1e-9

    def _polygon_boundary(self, triangles: Sequence[tuple[int, int, int]]) -> list[tuple[int, int]]:
        edge_counts: dict[tuple[int, int], int] = {}
        for a, b, c in triangles:
            for edge in ((a, b), (b, c), (c, a)):
                key = tuple(sorted(edge))
                edge_counts[key] = edge_counts.get(key, 0) + 1
        return [edge for edge, count in edge_counts.items() if count == 1]

    def _signed_area(self, triangle: tuple[int, int, int], all_points: np.ndarray) -> float:
        a, b, c = (all_points[index] for index in triangle)
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


class MinimumCurvatureOptimizer:
    """Smooth a centerline by reducing discrete second differences."""

    def __init__(
        self,
        iterations: int = 50,
        smooth_weight: float = 0.12,
        max_lateral_shift_m: float = 0.6,
    ) -> None:
        self.iterations = iterations
        self.smooth_weight = smooth_weight
        self.max_lateral_shift_m = max_lateral_shift_m

    def optimize(self, centerline: Sequence[Point2], closed: bool = True) -> list[Point2]:
        if len(centerline) < 4:
            return list(centerline)

        original = np.asarray(centerline, dtype=float)
        points = original.copy()
        movable = range(len(points)) if closed else range(1, len(points) - 1)

        for _ in range(self.iterations):
            updated = points.copy()
            for index in movable:
                prev_index = (index - 1) % len(points)
                next_index = (index + 1) % len(points)
                if not closed and (index == 0 or index == len(points) - 1):
                    continue

                target = 0.5 * (points[prev_index] + points[next_index])
                candidate = points[index] + self.smooth_weight * (target - points[index])
                offset = candidate - original[index]
                norm = float(np.linalg.norm(offset))
                if norm > self.max_lateral_shift_m:
                    candidate = original[index] + offset / norm * self.max_lateral_shift_m
                updated[index] = candidate
            points = updated

        return [(float(point[0]), float(point[1])) for point in points]


class DelaunayMinimumCurvaturePlanner:
    """Paper-style planner: landmarks -> Delaunay centerline -> smooth trajectory."""

    def __init__(
        self,
        expected_track_width_m: float = 6.0,
        waypoint_spacing_m: float = 2.0,
        target_speed_mps: float = 8.0,
    ) -> None:
        self.expected_track_width_m = expected_track_width_m
        self.waypoint_spacing_m = waypoint_spacing_m
        self.target_speed_mps = target_speed_mps
        self.triangulation = DelaunayTriangulation()
        self.optimizer = MinimumCurvatureOptimizer()

    def plan(
        self,
        landmarks: Sequence[ConeLandmark],
        start: Point2 | None = None,
        start_yaw_rad: float | None = None,
        closed: bool = True,
    ) -> PlannedPath:
        boundary_landmarks = [
            landmark
            for landmark in landmarks
            if landmark.color in {ConeColor.BLUE.value, ConeColor.YELLOW.value}
        ]
        points = [landmark.point for landmark in boundary_landmarks]
        triangles = self.triangulation.triangulate(points)
        cross_edges = self._delaunay_cross_edges(boundary_landmarks, triangles)
        raw_centerline = self._ordered_midpoints(boundary_landmarks, cross_edges, start, start_yaw_rad)

        if len(raw_centerline) < 4:
            raw_centerline = self._nearest_opposite_color_centerline(boundary_landmarks, start, start_yaw_rad)
        if len(raw_centerline) >= 2:
            raw_centerline = resample_polyline(raw_centerline, self.waypoint_spacing_m, closed=closed)

        trajectory = self.optimizer.optimize(raw_centerline, closed=closed)
        return PlannedPath(
            landmarks=list(boundary_landmarks),
            triangles=triangles,
            cross_edges=cross_edges,
            raw_centerline=raw_centerline,
            trajectory=trajectory,
            target_speed_mps=self.target_speed_mps,
        )

    def _delaunay_cross_edges(
        self,
        landmarks: Sequence[ConeLandmark],
        triangles: Sequence[tuple[int, int, int]],
    ) -> list[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        min_width = self.expected_track_width_m * 0.45
        max_width = self.expected_track_width_m * 1.8

        for triangle in triangles:
            for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                a, b = sorted(edge)
                if landmarks[a].color == landmarks[b].color:
                    continue
                width = self._distance(landmarks[a].point, landmarks[b].point)
                if min_width <= width <= max_width:
                    edges.add((a, b))

        return sorted(edges)

    def _ordered_midpoints(
        self,
        landmarks: Sequence[ConeLandmark],
        edges: Sequence[tuple[int, int]],
        start: Point2 | None,
        start_yaw_rad: float | None,
    ) -> list[Point2]:
        midpoints = [
            (
                (landmarks[a].x + landmarks[b].x) * 0.5,
                (landmarks[a].y + landmarks[b].y) * 0.5,
            )
            for a, b in edges
        ]
        return self._order_path(self._dedupe_points(midpoints), start, start_yaw_rad)

    def _nearest_opposite_color_centerline(
        self,
        landmarks: Sequence[ConeLandmark],
        start: Point2 | None,
        start_yaw_rad: float | None,
    ) -> list[Point2]:
        blue = [landmark for landmark in landmarks if landmark.color == ConeColor.BLUE.value]
        yellow = [landmark for landmark in landmarks if landmark.color == ConeColor.YELLOW.value]
        pairs: list[Point2] = []
        used_yellow: set[int] = set()

        for left in blue:
            candidates = [
                (index, right)
                for index, right in enumerate(yellow)
                if index not in used_yellow
            ]
            if not candidates:
                break
            index, right = min(candidates, key=lambda item: abs(self._distance(left.point, item[1].point) - self.expected_track_width_m))
            width = self._distance(left.point, right.point)
            if self.expected_track_width_m * 0.45 <= width <= self.expected_track_width_m * 1.8:
                used_yellow.add(index)
                pairs.append(((left.x + right.x) * 0.5, (left.y + right.y) * 0.5))

        return self._order_path(self._dedupe_points(pairs), start, start_yaw_rad)

    def _order_path(
        self,
        points: Sequence[Point2],
        start: Point2 | None,
        start_yaw_rad: float | None,
    ) -> list[Point2]:
        if len(points) < 2:
            return list(points)

        remaining = list(points)
        if start is None:
            centroid = self._centroid(remaining)
            first = min(remaining, key=lambda point: (point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2)
        else:
            first = min(remaining, key=lambda point: self._distance(point, start))

        ordered = [first]
        remaining.remove(first)
        heading = start_yaw_rad

        while remaining:
            current = ordered[-1]
            next_point = min(
                remaining,
                key=lambda point: self._extension_cost(current, point, heading),
            )
            if heading is not None:
                next_heading = math.atan2(next_point[1] - current[1], next_point[0] - current[0])
                heading = self._blend_angle(heading, next_heading, 0.7)
            else:
                heading = math.atan2(next_point[1] - current[1], next_point[0] - current[0])
            ordered.append(next_point)
            remaining.remove(next_point)

        return ordered

    def _extension_cost(self, current: Point2, candidate: Point2, heading: float | None) -> float:
        distance = self._distance(current, candidate)
        if heading is None:
            return distance
        candidate_heading = math.atan2(candidate[1] - current[1], candidate[0] - current[0])
        turn = abs(self._angle_delta(heading, candidate_heading))
        return distance + self.expected_track_width_m * turn

    def _dedupe_points(self, points: Sequence[Point2]) -> list[Point2]:
        result: list[Point2] = []
        for point in points:
            if all(self._distance(point, existing) > self.waypoint_spacing_m * 0.35 for existing in result):
                result.append(point)
        return result

    def _centroid(self, points: Sequence[Point2]) -> Point2:
        arr = np.asarray(points, dtype=float)
        center = arr.mean(axis=0)
        return (float(center[0]), float(center[1]))

    def _distance(self, a: Point2, b: Point2) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _angle_delta(self, a: float, b: float) -> float:
        delta = b - a
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta

    def _blend_angle(self, old: float, new: float, new_weight: float) -> float:
        return old + self._angle_delta(old, new) * new_weight


class LateralMPCController:
    """Sampling MPC for lateral path tracking with a kinematic bicycle model."""

    def __init__(
        self,
        wheelbase_m: float = 2.6,
        dt_s: float = 0.08,
        horizon_steps: int = 12,
        max_steer_rad: float = 0.55,
        target_speed_mps: float = 8.0,
    ) -> None:
        self.wheelbase_m = wheelbase_m
        self.dt_s = dt_s
        self.horizon_steps = horizon_steps
        self.max_steer_rad = max_steer_rad
        self.target_speed_mps = target_speed_mps

    def compute_control(
        self,
        state: VehicleState,
        trajectory: Sequence[Point2],
        closed: bool = True,
    ) -> ControlCommand:
        if len(trajectory) < 2:
            return ControlCommand(steer=0.0, throttle=0.0, brake=0.3, target_speed_mps=0.0)

        nearest = self._nearest_index(state, trajectory)
        candidates = np.linspace(-self.max_steer_rad, self.max_steer_rad, 31)
        best_steer = 0.0
        best_cost = float("inf")

        for steer in candidates:
            cost = self._rollout_cost(state, float(steer), trajectory, nearest, closed)
            if cost < best_cost:
                best_cost = cost
                best_steer = float(steer)

        speed_error = self.target_speed_mps - state.speed_mps
        throttle = max(0.0, min(1.0, 0.25 * speed_error))
        brake = max(0.0, min(1.0, -0.25 * speed_error))
        return ControlCommand(
            steer=max(-1.0, min(1.0, best_steer / self.max_steer_rad)),
            throttle=throttle,
            brake=brake,
            target_speed_mps=self.target_speed_mps,
        )

    def _rollout_cost(
        self,
        state: VehicleState,
        steer_rad: float,
        trajectory: Sequence[Point2],
        nearest_index: int,
        closed: bool,
    ) -> float:
        x = state.x
        y = state.y
        yaw = state.yaw_rad
        speed = max(state.speed_mps, 0.5)
        cost = 0.08 * steer_rad * steer_rad + 0.2 * (steer_rad - state.steer_rad) ** 2

        for step in range(1, self.horizon_steps + 1):
            x += speed * math.cos(yaw) * self.dt_s
            y += speed * math.sin(yaw) * self.dt_s
            yaw += speed / self.wheelbase_m * math.tan(steer_rad) * self.dt_s
            target_index = self._future_index(nearest_index, step, len(trajectory), closed)
            target = trajectory[target_index]
            target_yaw = self._path_heading(trajectory, target_index, closed)
            lateral_error = self._signed_lateral_error((x, y), target, target_yaw)
            heading_error = self._angle_delta(yaw, target_yaw)
            cost += 1.8 * lateral_error * lateral_error + 0.8 * heading_error * heading_error

        return cost

    def _nearest_index(self, state: VehicleState, trajectory: Sequence[Point2]) -> int:
        return min(
            range(len(trajectory)),
            key=lambda index: (trajectory[index][0] - state.x) ** 2 + (trajectory[index][1] - state.y) ** 2,
        )

    def _future_index(self, nearest_index: int, step: int, length: int, closed: bool) -> int:
        if closed:
            return (nearest_index + step) % length
        return min(nearest_index + step, length - 1)

    def _path_heading(self, trajectory: Sequence[Point2], index: int, closed: bool) -> float:
        a = trajectory[index]
        next_index = (index + 1) % len(trajectory) if closed else min(index + 1, len(trajectory) - 1)
        b = trajectory[next_index]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    def _signed_lateral_error(self, point: Point2, target: Point2, target_yaw: float) -> float:
        dx = point[0] - target[0]
        dy = point[1] - target[1]
        return -math.sin(target_yaw) * dx + math.cos(target_yaw) * dy

    def _angle_delta(self, a: float, b: float) -> float:
        delta = a - b
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        return delta


class WinningStackPlanner:
    """Facade for the simplified winning-stack pipeline used by demos/tests."""

    def __init__(
        self,
        expected_track_width_m: float = 6.0,
        waypoint_spacing_m: float = 2.0,
        target_speed_mps: float = 8.0,
    ) -> None:
        self.mapper = GraphSlamConeMapper()
        self.planner = DelaunayMinimumCurvaturePlanner(
            expected_track_width_m=expected_track_width_m,
            waypoint_spacing_m=waypoint_spacing_m,
            target_speed_mps=target_speed_mps,
        )
        self.controller = LateralMPCController(target_speed_mps=target_speed_mps)

    def plan_from_known_cones(
        self,
        cones: Iterable[Cone | ConeLandmark | dict],
        state: VehicleState,
        closed: bool = True,
    ) -> tuple[PlannedPath, ControlCommand]:
        landmarks = self.mapper.update_global_cones(self._track_boundary_cones(cones))
        path = self.planner.plan(
            landmarks,
            start=(state.x, state.y),
            start_yaw_rad=state.yaw_rad,
            closed=closed,
        )
        command = self.controller.compute_control(state, path.trajectory, closed=closed)
        return path, command

    def _track_boundary_cones(self, cones: Iterable[Cone | ConeLandmark | dict]) -> list[Cone | ConeLandmark | dict]:
        result: list[Cone | ConeLandmark | dict] = []
        for cone in cones:
            if isinstance(cone, Cone) and cone.role in {ConeRole.LEFT_BOUNDARY, ConeRole.RIGHT_BOUNDARY}:
                result.append(cone)
            elif not isinstance(cone, Cone):
                color = cone.color if isinstance(cone, ConeLandmark) else str(cone.get("color", ""))
                if color in {ConeColor.BLUE.value, ConeColor.YELLOW.value}:
                    result.append(cone)
        return result


def mean_nearest_error(reference: Sequence[Point2], estimate: Sequence[Point2]) -> float:
    if not reference or not estimate:
        return float("inf")

    total = 0.0
    for point in estimate:
        total += min(math.hypot(point[0] - ref[0], point[1] - ref[1]) for ref in reference)
    return total / len(estimate)


def path_curvature_energy(path: Sequence[Point2], closed: bool = True) -> float:
    if len(path) < 3:
        return 0.0

    points = np.asarray(path, dtype=float)
    indices = range(len(points)) if closed else range(1, len(points) - 1)
    energy = 0.0
    for index in indices:
        prev_point = points[(index - 1) % len(points)]
        point = points[index]
        next_point = points[(index + 1) % len(points)]
        a = point - prev_point
        b = next_point - point
        denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-6)
        energy += float(np.linalg.norm(b - a) / denom)
    length = max(polyline_length(path, closed=closed), 1e-6)
    return energy / length
