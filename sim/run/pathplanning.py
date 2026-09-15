from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

# None means no artificial top-speed target. The car still slows for curvature,
# braking distance, steering limits, and cone clearance.
MAX_SPEED_MPS: float | None = None
MIN_SPEED_MPS = 4.0
MAX_LATERAL_ACCEL_MPS2 = 10.0
MAX_ACCEL_MPS2 = 12.0
MAX_BRAKE_MPS2 = 18.0
MIN_SPEED_LOOKAHEAD_M = 8.0
MAX_SPEED_LOOKAHEAD_M = 35.0

import carla
from sim.env.blank_world import generate_blank_world
from sim.planning.paper_stack_planner import DelaunayTriangulation
from sim.roadgen import spawn_cones_in_carla
from sim.roadgen.geometry import Point2, resample_polyline
from sim.roadgen.track_generator import TrackGenerator


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw_rad: float
    speed_mps: float
    steer_rad: float


@dataclass(frozen=True)
class ReferencePoint:
    x_ref: float
    y_ref: float
    heading_ref: float
    curvature: float
    v_ref: float
    s_m: float


@dataclass(frozen=True)
class LateralCommand:
    steer: float
    steer_rad: float


class LongitudinalPID:
    """PID speed controller that maps speed error to CARLA throttle/brake."""

    def __init__(
        self,
        kp: float = 0.32,
        ki: float = 0.04,
        kd: float = 0.02,
        integral_limit: float = 8.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._last_error: float | None = None

    def control(
        self,
        target_speed_mps: float,
        current_speed_mps: float,
        dt_s: float,
    ) -> tuple[float, float]:
        # Step 1: keep the controller numerically stable if a CARLA tick is very short.
        dt_s = max(dt_s, 1e-3)

        # Step 2: an infinite target means "go as fast as possible here".
        if not math.isfinite(target_speed_mps):
            self._last_error = None
            return 1.0, 0.0

        # Step 3: compare the requested speed profile with the current vehicle speed.
        error = target_speed_mps - current_speed_mps

        # Step 4: accumulate low-frequency speed error so the car does not settle too slow.
        self._integral = clamp(
            self._integral + error * dt_s,
            -self.integral_limit,
            self.integral_limit,
        )

        # Step 5: damp sudden speed-error changes; this is intentionally small.
        derivative = 0.0
        if self._last_error is not None:
            derivative = (error - self._last_error) / dt_s
        self._last_error = error

        # Step 6: convert PID effort into either throttle or brake.
        effort = self.kp * error + self.ki * self._integral + self.kd * derivative
        if effort >= 0.0:
            return clamp(effort, 0.0, 1.0), 0.0
        return 0.0, clamp(-effort, 0.0, 1.0)


class LateralMPCController:
    """Small sampling MPC using a kinematic bicycle rollout."""

    def __init__(
        self,
        wheelbase_m: float = 2.875,
        dt_s: float = 0.08,
        horizon_steps: int = 16,
        max_steer_rad: float = 0.55,
        steering_samples: int = 41,
        cone_clearance_m: float = 1.10,
    ) -> None:
        self.wheelbase_m = wheelbase_m
        self.dt_s = dt_s
        self.horizon_steps = horizon_steps
        self.max_steer_rad = max_steer_rad
        self.steering_samples = steering_samples
        self.cone_clearance_m = cone_clearance_m

    def control(
        self,
        state: VehicleState,
        trajectory: Sequence[ReferencePoint],
        cone_points: Sequence[Point2],
        closed: bool,
    ) -> LateralCommand:
        # Step 1: fail safe if the planner has not produced enough path points.
        if len(trajectory) < 2:
            return LateralCommand(steer=0.0, steer_rad=0.0)

        # Step 2: anchor the receding-horizon rollout at the closest reference point.
        nearest_index = nearest_reference_index(trajectory, state.x, state.y)

        # Step 3: sample steering candidates across the allowed steering range.
        candidates = [
            -self.max_steer_rad
            + 2.0 * self.max_steer_rad * i / max(self.steering_samples - 1, 1)
            for i in range(self.steering_samples)
        ]
        best_steer = 0.0
        best_cost = float("inf")

        # Step 4: predict each candidate and keep the one with the lowest tracking cost.
        for steer_rad in candidates:
            cost = self._rollout_cost(
                state=state,
                steer_rad=steer_rad,
                trajectory=trajectory,
                cone_points=cone_points,
                nearest_index=nearest_index,
                closed=closed,
            )
            if cost < best_cost:
                best_cost = cost
                best_steer = steer_rad

        # Step 5: CARLA expects normalized steering in [-1, 1].
        return LateralCommand(
            steer=clamp(best_steer / self.max_steer_rad, -1.0, 1.0),
            steer_rad=best_steer,
        )

    def _rollout_cost(
        self,
        state: VehicleState,
        steer_rad: float,
        trajectory: Sequence[ReferencePoint],
        cone_points: Sequence[Point2],
        nearest_index: int,
        closed: bool,
    ) -> float:
        # Step 1: start the prediction from the measured vehicle state.
        x = state.x
        y = state.y
        yaw = state.yaw_rad
        speed = max(state.speed_mps, 0.5)
        distance_ahead = 0.0

        # Step 2: discourage large steering and abrupt steering changes.
        cost = 0.12 * steer_rad * steer_rad + 0.65 * (steer_rad - state.steer_rad) ** 2

        for _ in range(self.horizon_steps):
            # Step 3: roll the kinematic bicycle model forward one MPC timestep.
            x += speed * math.cos(yaw) * self.dt_s
            y += speed * math.sin(yaw) * self.dt_s
            yaw = wrap_angle(
                yaw + speed / self.wheelbase_m * math.tan(steer_rad) * self.dt_s
            )
            distance_ahead += speed * self.dt_s

            # Step 4: compare the predicted vehicle pose with the matching future reference.
            ref_index = reference_index_ahead(
                trajectory=trajectory,
                start_index=nearest_index,
                distance_ahead_m=distance_ahead,
                closed=closed,
            )
            ref = trajectory[ref_index]
            lateral_error = signed_lateral_error(x, y, ref)
            heading_error = wrap_angle(yaw - ref.heading_ref)
            cost += (
                3.0 * lateral_error * lateral_error
                + 1.2 * heading_error * heading_error
            )

            # Step 5: heavily penalize rollouts that pass too close to cones.
            cost += self._cone_clearance_cost(x, y, cone_points)

        return cost

    def _cone_clearance_cost(
        self,
        x: float,
        y: float,
        cone_points: Sequence[Point2],
    ) -> float:
        if not cone_points:
            return 0.0
        nearest = min(math.hypot(x - cx, y - cy) for cx, cy in cone_points)
        if nearest >= self.cone_clearance_m:
            return 0.0
        margin_error = self.cone_clearance_m - nearest
        return 500.0 + 800.0 * margin_error * margin_error


def spawn_one_cone(
    world: carla.World,
    x: float = 5.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
) -> carla.Actor:
    """Spawn one CARLA built-in traffic cone at a world position."""
    blueprint = world.get_blueprint_library().find("static.prop.trafficcone01")
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=yaw),
    )
    return world.spawn_actor(blueprint, transform)


def setup_client() -> carla.Client:
    """Connect to the CARLA server and return a configured client."""
    client = carla.Client("127.0.0.1", 2000)
    # Generating an OpenDRIVE world can take longer than a normal RPC call.
    client.set_timeout(120.0)
    return client


def setup_world(client: carla.Client) -> carla.World:
    """Generate the project's flat test map."""
    return generate_blank_world(
        client,
        name="FSAEBlank",
        length_m=320.0,
        lane_width_m=140.0,
        additional_width_m=20.0,
    )


def spawn_car(
    world: carla.World,
    x: float,
    y: float,
    heading_rad: float,
) -> carla.Vehicle:
    """Spawn the ego vehicle at the generated Autocross start pose."""
    blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
    blueprint.set_attribute("role_name", "ego")
    transform = carla.Transform(
        carla.Location(x=x, y=y, z=0.4),
        carla.Rotation(yaw=heading_rad * 180.0 / 3.141592653589793),
    )
    vehicle = world.try_spawn_actor(blueprint, transform)
    if vehicle is None:
        raise RuntimeError(f"Could not spawn vehicle at ({x:.2f}, {y:.2f}).")
    vehicle.set_autopilot(False)
    return vehicle


def get_vehicle_pose(vehicle: carla.Vehicle) -> dict[str, float]:
    """Read the vehicle's current CARLA world pose."""
    transform = vehicle.get_transform()
    return {
        "x": transform.location.x,
        "y": transform.location.y,
        "z": transform.location.z,
        "yaw": transform.rotation.yaw,
    }


def get_cone_positions(cone_actors: list[carla.Actor]) -> list[dict[str, float]]:
    """Read the current CARLA world pose of every spawned cone."""
    positions: list[dict[str, float]] = []
    for cone in cone_actors:
        transform = cone.get_transform()
        positions.append(
            {
                "x": transform.location.x,
                "y": transform.location.y,
                "z": transform.location.z,
                "yaw": transform.rotation.yaw,
            }
        )
    return positions


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def build_reference_trajectory(
    centerline: Sequence[Point2],
    closed: bool,
    spacing_m: float = 1.0,
    min_speed_mps: float = 3.0,
    max_speed_mps: float | None = 8.0,
    max_lateral_accel_mps2: float = 4.0,
    max_accel_mps2: float = 2.2,
    max_brake_mps2: float = 5.0,
) -> list[ReferencePoint]:
    # Step 1: resample the centerline so every controller step sees even spacing.
    points = resample_polyline(centerline, spacing_m=spacing_m, closed=closed)
    if len(points) < 3:
        raise ValueError("At least three centerline points are required for control.")

    # Step 2: compute arc length, heading, and curvature for every reference point.
    s_values = cumulative_distances(points, closed=False)
    headings = [
        reference_heading(points, index, closed=closed) for index in range(len(points))
    ]
    curvatures = [
        reference_curvature(points, index, closed=closed)
        for index in range(len(points))
    ]

    # Step 3: convert curvature into a safe target speed using lateral acceleration.
    speeds = [
        curvature_limited_speed(
            curvature=curvature,
            min_speed_mps=min_speed_mps,
            max_speed_mps=max_speed_mps,
            max_lateral_accel_mps2=max_lateral_accel_mps2,
        )
        for curvature in curvatures
    ]

    # Step 4: make the speed profile physically reachable before and after corners.
    speeds = apply_acceleration_limits(
        speeds=speeds,
        points=points,
        closed=closed,
        max_accel_mps2=max_accel_mps2,
        max_brake_mps2=max_brake_mps2,
    )

    # Step 5: pack the controller reference as x/y, heading, curvature, speed, and s.
    return [
        ReferencePoint(
            x_ref=x,
            y_ref=y,
            heading_ref=headings[index],
            curvature=curvatures[index],
            v_ref=speeds[index],
            s_m=s_values[index],
        )
        for index, (x, y) in enumerate(points)
    ]


def cumulative_distances(points: Sequence[Point2], closed: bool) -> list[float]:
    # Step 1: accumulate distance along each segment of the reference path.
    distances = [0.0]
    for first, second in zip(points, points[1:]):
        distances.append(
            distances[-1] + math.hypot(second[0] - first[0], second[1] - first[1])
        )

    # Step 2: include the closing segment when a closed-loop length is needed.
    if closed and len(points) > 1:
        distances.append(
            distances[-1]
            + math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1])
        )
    return distances[: len(points)]


def reference_heading(points: Sequence[Point2], index: int, closed: bool) -> float:
    # Step 1: use neighboring points to estimate the local tangent direction.
    if closed:
        prev_point = points[(index - 1) % len(points)]
        next_point = points[(index + 1) % len(points)]
    else:
        prev_point = points[max(index - 1, 0)]
        next_point = points[min(index + 1, len(points) - 1)]

    # Step 2: convert the tangent vector into a world-frame heading angle.
    return math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])


def reference_curvature(points: Sequence[Point2], index: int, closed: bool) -> float:
    # Step 1: choose a three-point stencil around the current reference point.
    if closed:
        prev_point = points[(index - 1) % len(points)]
        point = points[index]
        next_point = points[(index + 1) % len(points)]
    elif index == 0 or index == len(points) - 1:
        return 0.0
    else:
        prev_point = points[index - 1]
        point = points[index]
        next_point = points[index + 1]

    # Step 2: compute signed curvature from the triangle formed by the three points.
    ax = point[0] - prev_point[0]
    ay = point[1] - prev_point[1]
    bx = next_point[0] - point[0]
    by = next_point[1] - point[1]
    cx = next_point[0] - prev_point[0]
    cy = next_point[1] - prev_point[1]
    cross = ax * by - ay * bx
    denom = math.hypot(ax, ay) * math.hypot(bx, by) * math.hypot(cx, cy)
    if denom < 1e-6:
        return 0.0
    return 2.0 * cross / denom


def curvature_limited_speed(
    curvature: float,
    min_speed_mps: float,
    max_speed_mps: float | None,
    max_lateral_accel_mps2: float,
) -> float:
    # Step 1: straight or nearly straight segments may use the configured top speed.
    abs_curvature = abs(curvature)
    if abs_curvature < 1e-4:
        return math.inf if max_speed_mps is None else max_speed_mps

    # Step 2: use v = sqrt(a_lat / curvature) to keep lateral acceleration bounded.
    curve_speed = math.sqrt(max_lateral_accel_mps2 / abs_curvature)
    if max_speed_mps is None:
        return max(min_speed_mps, curve_speed)
    return clamp(curve_speed, min_speed_mps, max_speed_mps)


def apply_acceleration_limits(
    speeds: Sequence[float],
    points: Sequence[Point2],
    closed: bool,
    max_accel_mps2: float,
    max_brake_mps2: float,
) -> list[float]:
    limited = list(speeds)
    if len(limited) < 2:
        return limited

    passes = 3 if closed else 1
    for _ in range(passes):
        # Step 1: forward pass limits how quickly the car can accelerate.
        for index in range(len(limited) - 1):
            ds = distance_2d(points[index], points[index + 1])
            limited[index + 1] = min(
                limited[index + 1],
                math.sqrt(max(limited[index] ** 2 + 2.0 * max_accel_mps2 * ds, 0.0)),
            )

        # Step 2: wrap the forward pass across the start line for closed tracks.
        if closed:
            ds = distance_2d(points[-1], points[0])
            limited[0] = min(
                limited[0],
                math.sqrt(max(limited[-1] ** 2 + 2.0 * max_accel_mps2 * ds, 0.0)),
            )

        # Step 3: backward pass limits speed before corners so braking is feasible.
        for index in range(len(limited) - 1, 0, -1):
            ds = distance_2d(points[index - 1], points[index])
            limited[index - 1] = min(
                limited[index - 1],
                math.sqrt(max(limited[index] ** 2 + 2.0 * max_brake_mps2 * ds, 0.0)),
            )

        # Step 4: wrap the braking pass across the start line for closed tracks.
        if closed:
            ds = distance_2d(points[-1], points[0])
            limited[-1] = min(
                limited[-1],
                math.sqrt(max(limited[0] ** 2 + 2.0 * max_brake_mps2 * ds, 0.0)),
            )

    return limited


def nearest_reference_index(
    trajectory: Sequence[ReferencePoint],
    x: float,
    y: float,
) -> int:
    return min(
        range(len(trajectory)),
        key=lambda index: (trajectory[index].x_ref - x) ** 2
        + (trajectory[index].y_ref - y) ** 2,
    )


def reference_index_ahead(
    trajectory: Sequence[ReferencePoint],
    start_index: int,
    distance_ahead_m: float,
    closed: bool,
) -> int:
    target_s = trajectory[start_index].s_m + distance_ahead_m
    total_length = trajectory[-1].s_m + distance_2d(
        (trajectory[-1].x_ref, trajectory[-1].y_ref),
        (trajectory[0].x_ref, trajectory[0].y_ref),
    )
    if closed:
        target_s %= total_length
        for index, ref in enumerate(trajectory):
            if ref.s_m >= target_s:
                return index
        return 0

    if target_s >= trajectory[-1].s_m:
        return len(trajectory) - 1
    for index in range(start_index, len(trajectory)):
        if trajectory[index].s_m >= target_s:
            return index
    return len(trajectory) - 1


def signed_lateral_error(x: float, y: float, ref: ReferencePoint) -> float:
    dx = x - ref.x_ref
    dy = y - ref.y_ref
    return -math.sin(ref.heading_ref) * dx + math.cos(ref.heading_ref) * dy


def distance_2d(first: Point2, second: Point2) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def target_speed_ahead(
    trajectory: Sequence[ReferencePoint],
    nearest_index: int,
    lookahead_m: float,
    closed: bool,
) -> float:
    end_index = reference_index_ahead(
        trajectory=trajectory,
        start_index=nearest_index,
        distance_ahead_m=lookahead_m,
        closed=closed,
    )
    if closed and end_index < nearest_index:
        refs = list(trajectory[nearest_index:]) + list(trajectory[: end_index + 1])
    else:
        refs = list(trajectory[nearest_index : end_index + 1])
    return min(ref.v_ref for ref in refs) if refs else trajectory[nearest_index].v_ref


def read_vehicle_state(vehicle: carla.Vehicle, last_steer_rad: float) -> VehicleState:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    speed = math.sqrt(
        velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
    )
    return VehicleState(
        x=transform.location.x,
        y=transform.location.y,
        yaw_rad=math.radians(transform.rotation.yaw),
        speed_mps=speed,
        steer_rad=last_steer_rad,
    )


def draw_reference_trajectory(
    world: carla.World,
    trajectory: Sequence[ReferencePoint],
    lifetime_s: float = 0.0,
) -> None:
    finite_speeds = [ref.v_ref for ref in trajectory if math.isfinite(ref.v_ref)]
    draw_speed_scale = max(finite_speeds, default=MIN_SPEED_MPS)
    if MAX_SPEED_MPS is not None:
        draw_speed_scale = MAX_SPEED_MPS

    for first, second in zip(trajectory, trajectory[1:]):
        speed_ratio = 1.0
        if math.isfinite(first.v_ref) and draw_speed_scale > 0.0:
            speed_ratio = clamp(first.v_ref / draw_speed_scale, 0.0, 1.0)
        world.debug.draw_line(
            carla.Location(x=first.x_ref, y=first.y_ref, z=0.32),
            carla.Location(x=second.x_ref, y=second.y_ref, z=0.32),
            thickness=0.07,
            color=carla.Color(
                int(255 * (1.0 - speed_ratio)),
                int(255 * speed_ratio),
                40,
            ),
            life_time=lifetime_s,
        )
    if trajectory:
        first = trajectory[-1]
        second = trajectory[0]
        world.debug.draw_line(
            carla.Location(x=first.x_ref, y=first.y_ref, z=0.32),
            carla.Location(x=second.x_ref, y=second.y_ref, z=0.32),
            thickness=0.07,
            color=carla.Color(80, 220, 80),
            life_time=lifetime_s,
        )


def format_speed(speed_mps: float) -> str:
    if not math.isfinite(speed_mps):
        return "unlimited"
    return f"{speed_mps:.2f}"


def render_delaunay_midpoints(
    world: carla.World,
    track,
    lifetime_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Triangulate boundary cones and draw cross-track midpoint candidates."""
    # Orange cones mark the start/finish area; they are not lane boundaries.
    cones = [cone for cone in track.cones if cone.color.value in {"blue", "yellow"}]
    points = [(cone.x, cone.y) for cone in cones]
    triangles = DelaunayTriangulation().triangulate(points)

    edge_set: set[tuple[int, int]] = set()
    for a, b, c in triangles:
        edge_set.update(
            {tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))}
        )

    # Draw the complete Delaunay graph in gray.
    for a, b in edge_set:
        world.debug.draw_line(
            carla.Location(x=points[a][0], y=points[a][1], z=0.12),
            carla.Location(x=points[b][0], y=points[b][1], z=0.12),
            thickness=0.01,
            color=carla.Color(120, 120, 120),
            life_time=lifetime_s,
        )

    midpoint_candidates: list[tuple[float, float]] = []
    min_width = track.width * 0.45
    max_width = track.width * 1.8
    for a, b in sorted(edge_set):
        first = cones[a]
        second = cones[b]
        if first.color.value == second.color.value:
            continue
        width = math.hypot(first.x - second.x, first.y - second.y)
        if not min_width <= width <= max_width:
            continue

        midpoint = ((first.x + second.x) * 0.5, (first.y + second.y) * 0.5)
        midpoint_candidates.append(midpoint)
        world.debug.draw_line(
            carla.Location(x=first.x, y=first.y, z=0.18),
            carla.Location(x=second.x, y=second.y, z=0.18),
            thickness=0.05,
            color=carla.Color(0, 220, 0),
            life_time=lifetime_s,
        )

    # Connect midpoint candidates in their generation order as a first visual path.
    for first, second in zip(midpoint_candidates, midpoint_candidates[1:]):
        world.debug.draw_line(
            carla.Location(x=first[0], y=first[1], z=0.22),
            carla.Location(x=second[0], y=second[1], z=0.22),
            thickness=0.08,
            color=carla.Color(220, 40, 40),
            life_time=lifetime_s,
        )

    print(
        f"[delaunay] cones={len(points)} triangles={len(triangles)} "
        f"edges={len(edge_set)} cross_edges={len(midpoint_candidates)}"
    )
    return midpoint_candidates


# def drive_straight(
#     world: carla.World,
#     vehicle: carla.Vehicle,
#     duration_s: float = 5.0,
# ) -> None:
#     """Drive forward with a fixed throttle, then stop the vehicle."""
#     end_time = time.monotonic() + duration_s
#     while time.monotonic() < end_time:
#         vehicle.apply_control(
#             carla.VehicleControl(
#                 throttle=0.25,
#                 steer=0.0,
#                 brake=0.0,
#                 hand_brake=False,
#                 reverse=False,
#             )
#         )
#         world.wait_for_tick()

#     vehicle.apply_control(
#         carla.VehicleControl(
#             throttle=0.0,
#             steer=0.0,
#             brake=1.0,
#             hand_brake=False,
#             reverse=False,
#         )
#     )


def generate_autocross_track(
    client: carla.Client,
    name: str = "my_autocross",
    length: float = 260.0,
    width: float = 6.0,
    curvature: float = 0.35,
    cone_density: float = 0.65,
) -> tuple[carla.World, list[carla.Actor]]:
    """Generate an autocross track in the CARLA world and spawn cones."""
    world = setup_world(client)
    generator = TrackGenerator(seed=42)
    track = generator.generate_autocross(
        name=name,
        length=length,
        width=width,
        curvature=curvature,
        cone_density=cone_density,
    )
    mesh = generator.build_mesh_spec(track)
    carla_spec = generator.build_carla_import_spec(track, mesh)
    cone_actors = spawn_cones_in_carla(world, carla_spec)
    start_x, start_y, start_heading = carla_spec.start_pose
    spawn_car(world, start_x, start_y, start_heading)
    return world, cone_actors


def main() -> None:
    client = setup_client()
    print("[step-1] connected to CARLA at 127.0.0.1:2000")

    world = setup_world(client)
    print(f"[step-1] world ready: {world.get_map().name}")
    print(f"[step-1] map spawn points: {len(world.get_map().get_spawn_points())}")

    generator = TrackGenerator(seed=42)
    track = generator.generate_autocross(
        name="my_autocross",
        length=260.0,
        width=6.0,
        curvature=0.35,
        cone_density=0.65,
    )
    mesh = generator.build_mesh_spec(track)
    carla_spec = generator.build_carla_import_spec(track, mesh)
    cone_actors = spawn_cones_in_carla(world, carla_spec)
    start_x, start_y, start_heading = carla_spec.start_pose
    car = spawn_car(world, start_x, start_y, start_heading)

    cone_positions = get_cone_positions(cone_actors)
    print(f"[step-5] first cone poses: {cone_positions[:5]}")

    midpoint_candidates = render_delaunay_midpoints(world, track)
    print(f"[step-6] midpoint candidates: {len(midpoint_candidates)}")

    # Step 1: turn the known centerline into controller-ready reference points.
    reference_trajectory = build_reference_trajectory(
        centerline=track.centerline,
        closed=track.closed,
        spacing_m=1.0,
        min_speed_mps=MIN_SPEED_MPS,
        max_speed_mps=MAX_SPEED_MPS,
        max_lateral_accel_mps2=MAX_LATERAL_ACCEL_MPS2,
        max_accel_mps2=MAX_ACCEL_MPS2,
        max_brake_mps2=MAX_BRAKE_MPS2,
    )

    # Step 2: draw the reference trajectory; greener segments allow higher speed.
    draw_reference_trajectory(world, reference_trajectory)

    # Step 3: collect cone positions for the MPC clearance penalty.
    cone_points = [(cone.x, cone.y) for cone in track.cones]

    # Step 4: create one lateral controller and one longitudinal controller.
    lateral_mpc = LateralMPCController()
    longitudinal_pid = LongitudinalPID()

    # Step 5: keep lightweight controller memory between ticks.
    last_steer_rad = 0.0
    last_time = time.monotonic()
    last_print_time = last_time
    print(f"[step-7] reference trajectory points: {len(reference_trajectory)}")
    print("[drive] running controller loop; press Ctrl+C to stop")

    try:
        while True:
            # Step 6: measure elapsed time for the PID controller.
            now = time.monotonic()
            dt_s = clamp(now - last_time, 1e-3, 0.2)
            last_time = now

            # Step 7: read the vehicle pose, heading, speed, and previous steer angle.
            state = read_vehicle_state(car, last_steer_rad)

            # Step 8: locate the nearest point on the reference trajectory.
            nearest_index = nearest_reference_index(
                reference_trajectory,
                state.x,
                state.y,
            )
            print(nearest_index, len(reference_trajectory), state.x, state.y)
            # Step 9: look ahead along the speed profile and use the slowest upcoming speed.
            speed_lookahead_m = clamp(
                state.speed_mps * 1.4 + 6.0,
                MIN_SPEED_LOOKAHEAD_M,
                MAX_SPEED_LOOKAHEAD_M,
            )
            target_speed_mps = target_speed_ahead(
                trajectory=reference_trajectory,
                nearest_index=nearest_index,
                lookahead_m=speed_lookahead_m,
                closed=track.closed,
            )

            # Step 10: lateral MPC chooses the steering command.
            lateral_command = lateral_mpc.control(
                state=state,
                trajectory=reference_trajectory,
                cone_points=cone_points,
                closed=track.closed,
            )

            # Step 11: longitudinal PID chooses throttle or brake for the target speed.
            throttle, brake = longitudinal_pid.control(
                target_speed_mps=target_speed_mps,
                current_speed_mps=state.speed_mps,
                dt_s=dt_s,
            )

            # Step 12: send the combined steering and speed command to CARLA.
            car.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=lateral_command.steer,
                    brake=brake,
                    hand_brake=False,
                    reverse=False,
                )
            )

            # Step 13: remember the physical steering angle for steering-rate cost.
            last_steer_rad = lateral_command.steer_rad

            # Step 14: print compact runtime diagnostics once per second.
            if now - last_print_time >= 1.0:
                ref = reference_trajectory[nearest_index]
                cross_track_error = signed_lateral_error(state.x, state.y, ref)
                print(
                    "[drive] "
                    f"speed={state.speed_mps:.2f}/{format_speed(target_speed_mps)} m/s "
                    f"steer={lateral_command.steer:+.2f} "
                    f"thr={throttle:.2f} brake={brake:.2f} "
                    f"cte={cross_track_error:+.2f} m "
                    f"kappa={ref.curvature:+.3f}"
                )
                last_print_time = now

            # Step 15: wait for the simulator to advance before computing again.
            world.wait_for_tick()
    except KeyboardInterrupt:
        # Step 16: if the user stops the run, command a full brake.
        car.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
                reverse=False,
            )
        )
        print("[drive] stopped")


if __name__ == "__main__":
    main()
