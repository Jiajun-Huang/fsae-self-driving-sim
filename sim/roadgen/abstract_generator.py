from __future__ import annotations

import math

import numpy as np

from .geometry import (
    Point2,
    catmull_rom_closed,
    center_points,
    headings_and_normals,
    offset_polyline,
    polyline_length,
    resample_polyline,
    scale_points,
)
from .models import AbstractTrack, Cone, LineMark
from .rules import (
    DEFAULT_RULES,
    LARGE_CONE,
    SMALL_CONE,
    ConeColor,
    ConeRole,
    FSAERuleProfile,
    TrackType,
)


class AbstractTrackGenerator:
    """Generates rule-aware track geometry without CARLA-specific objects."""

    def __init__(
        self,
        seed: int | None = None,
        rules: FSAERuleProfile = DEFAULT_RULES,
    ) -> None:
        self.rules = rules
        self.rng = np.random.default_rng(seed)

    def generate_autocross(
        self,
        name: str = "autocross",
        length_m: float = 260.0,
        width_m: float | None = None,
        curvature: float = 0.35,
        cone_density: float = 0.65,
    ) -> AbstractTrack:
        width = self._validated_width(width_m)
        self._validate_autocross_length(length_m)
        centerline = self._random_closed_centerline(length_m, curvature)
        left_boundary, right_boundary = offset_polyline(centerline, width, closed=True)
        headings, normals = headings_and_normals(centerline, closed=True)

        cones = self._boundary_cones(
            left_boundary,
            right_boundary,
            closed=True,
            cone_density=cone_density,
        )
        start_line = LineMark(
            center=centerline[0],
            heading_rad=headings[0],
            width_m=width,
            kind="start_finish",
        )
        cones.extend(self._start_finish_cones(start_line, normals[0]))
        cones.extend(self._entry_lane_cones(start_line, normals[0]))

        return AbstractTrack(
            name=name,
            track_type=TrackType.AUTOCROSS,
            centerline=centerline,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            cones=cones,
            width_m=width,
            closed=True,
            start_line=start_line,
            finish_line=start_line,
            metadata={
                "rules": "FSAE Driverless 2026 DD.1 and DD.4.4",
                "max_straight_m": self.rules.max_autocross_straight_m,
                "min_turning_diameter_m": self.rules.min_turning_diameter_m,
                "stop_after_finish_m": self.rules.autocross_stop_length_m,
            },
        )

    def generate_acceleration(
        self,
        name: str = "acceleration",
        width_m: float | None = None,
        cone_density: float = 0.65,
    ) -> AbstractTrack:
        width = self._validated_width(width_m)
        course_len = self.rules.acceleration_course_length_m
        stop_len = self.rules.acceleration_stop_length_m
        centerline = [(-10.0, 0.0), (0.0, 0.0), (course_len, 0.0), (course_len + stop_len, 0.0)]
        centerline = resample_polyline(centerline, spacing_m=2.0, closed=False)
        left_boundary, right_boundary = offset_polyline(centerline, width, closed=False)
        cones = self._boundary_cones(
            left_boundary,
            right_boundary,
            closed=False,
            cone_density=cone_density,
        )
        start_line = LineMark(center=(0.0, 0.0), heading_rad=0.0, width_m=width, kind="start")
        finish_line = LineMark(
            center=(course_len, 0.0),
            heading_rad=0.0,
            width_m=width,
            kind="finish",
        )
        road_normal = (0.0, 1.0)
        cones.extend(self._start_finish_cones(start_line, road_normal))
        cones.extend(self._start_finish_cones(finish_line, road_normal))

        return AbstractTrack(
            name=name,
            track_type=TrackType.ACCELERATION,
            centerline=centerline,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            cones=cones,
            width_m=width,
            closed=False,
            start_line=start_line,
            finish_line=finish_line,
            metadata={
                "rules": "FSAE Driverless 2026 DD.1 and DD.4.2",
                "timed_length_m": course_len,
                "stop_after_finish_m": stop_len,
                "staging_offset_m": self.rules.acceleration_staging_offset_m,
            },
        )

    def generate_skidpad(
        self,
        name: str = "skidpad",
        width_m: float | None = None,
        cone_density: float = 0.85,
    ) -> AbstractTrack:
        width = self._validated_width(
            width_m if width_m is not None else self.rules.min_track_width_m
        )
        inner_radius = self.rules.skidpad_inner_radius_m
        outer_radius = self.rules.skidpad_outer_radius_m
        center_radius = (inner_radius + outer_radius) / 2.0
        half_center_distance = self.rules.skidpad_center_distance_m / 2.0
        left_center = (-half_center_distance, 0.0)
        right_center = (half_center_distance, 0.0)

        centerline = self._skidpad_centerline(left_center, right_center, center_radius)
        left_boundary, right_boundary = offset_polyline(centerline, width, closed=True)
        headings, normals = headings_and_normals(centerline, closed=True)

        cones = self._skidpad_cones(
            left_center=left_center,
            right_center=right_center,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            cone_density=cone_density,
        )
        start_line = LineMark(
            center=centerline[0],
            heading_rad=headings[0],
            width_m=width,
            kind="start_finish",
        )
        cones.extend(self._start_finish_cones(start_line, normals[0]))

        return AbstractTrack(
            name=name,
            track_type=TrackType.SKIDPAD,
            centerline=centerline,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            cones=cones,
            width_m=width,
            closed=True,
            start_line=start_line,
            finish_line=start_line,
            metadata={
                "rules": "FSAE Driverless 2026 DD.1 and DD.4.3",
                "inner_cones_per_circle": self.rules.skidpad_inner_cones_per_circle,
                "inner_radius_m": inner_radius,
                "outer_radius_m": outer_radius,
                "stop_after_finish_m": self.rules.skidpad_stop_length_m,
            },
        )

    def _random_closed_centerline(
        self,
        length_m: float,
        curvature: float,
    ) -> list[Point2]:
        curvature = float(np.clip(curvature, 0.0, 0.6))
        control_count = max(14, int(length_m / 22.0))
        base_radius = length_m / (2.0 * math.pi)
        base_angles = np.linspace(0.0, 2.0 * math.pi, control_count, endpoint=False)
        angle_jitter = self.rng.uniform(
            -0.35 / control_count,
            0.35 / control_count,
            size=control_count,
        )
        angles = base_angles + angle_jitter
        phase_a = self.rng.uniform(0.0, 2.0 * math.pi)
        phase_b = self.rng.uniform(0.0, 2.0 * math.pi)
        phase_c = self.rng.uniform(0.0, 2.0 * math.pi)
        phase_d = self.rng.uniform(0.0, 2.0 * math.pi)
        harmonic = (
            0.55 * curvature * np.sin(2.0 * angles + phase_a)
            + 0.35 * curvature * np.sin(3.0 * angles + phase_b)
        )
        radial_noise = self.rng.uniform(-0.45 * curvature, 0.45 * curvature, size=control_count)
        radial_noise = radial_noise + harmonic
        radial_noise = np.convolve(
            np.r_[radial_noise[-1], radial_noise, radial_noise[0]],
            np.ones(3) / 3.0,
            mode="valid",
        )
        x_scale = 1.0
        y_scale = self.rng.uniform(0.68, 0.92)
        controls = [
            (
                base_radius
                * x_scale
                * float(np.clip(1.0 + radial_noise[i], 0.62, 1.45))
                * math.cos(angle)
                + base_radius * curvature * 0.52 * math.sin(3.0 * angle + phase_c),
                base_radius
                * y_scale
                * float(np.clip(1.0 + radial_noise[i], 0.62, 1.45))
                * math.sin(angle)
                + base_radius * curvature * 0.42 * math.sin(2.0 * angle + phase_d),
            )
            for i, angle in enumerate(angles)
        ]
        smooth = center_points(catmull_rom_closed(controls, samples_per_segment=18))
        current_len = polyline_length(smooth, closed=True)
        scaled = scale_points(smooth, length_m / current_len)
        return resample_polyline(scaled, spacing_m=2.0, closed=True)

    def _skidpad_centerline(
        self,
        left_center: Point2,
        right_center: Point2,
        radius_m: float,
    ) -> list[Point2]:
        samples_per_circle = 120
        right_angles = np.linspace(math.pi, -math.pi, samples_per_circle, endpoint=False)
        left_angles = np.linspace(0.0, 2.0 * math.pi, samples_per_circle, endpoint=False)
        points: list[Point2] = []
        for angle in right_angles:
            points.append(
                (
                    right_center[0] + radius_m * math.cos(angle),
                    right_center[1] + radius_m * math.sin(angle),
                )
            )
        for angle in left_angles:
            points.append(
                (
                    left_center[0] + radius_m * math.cos(angle),
                    left_center[1] + radius_m * math.sin(angle),
                )
            )
        return resample_polyline(points, spacing_m=1.0, closed=True)

    def _boundary_cones(
        self,
        left_boundary: list[Point2],
        right_boundary: list[Point2],
        closed: bool,
        cone_density: float,
    ) -> list[Cone]:
        spacing = self._cone_spacing(cone_density)
        left_points = resample_polyline(left_boundary, spacing_m=spacing, closed=closed)
        right_points = resample_polyline(right_boundary, spacing_m=spacing, closed=closed)
        cones = [
            Cone(
                x=x,
                y=y,
                color=ConeColor.BLUE,
                size=SMALL_CONE,
                role=ConeRole.LEFT_BOUNDARY,
            )
            for x, y in left_points
        ]
        cones.extend(
            Cone(
                x=x,
                y=y,
                color=ConeColor.YELLOW,
                size=SMALL_CONE,
                role=ConeRole.RIGHT_BOUNDARY,
            )
            for x, y in right_points
        )
        return cones

    def _skidpad_cones(
        self,
        left_center: Point2,
        right_center: Point2,
        inner_radius: float,
        outer_radius: float,
        cone_density: float,
    ) -> list[Cone]:
        cones: list[Cone] = []
        inner_count = self.rules.skidpad_inner_cones_per_circle
        outer_count = max(24, int((2.0 * math.pi * outer_radius) / self._cone_spacing(cone_density)))

        cones.extend(
            self._circle_cones(
                center=left_center,
                radius_m=inner_radius,
                count=inner_count,
                color=ConeColor.BLUE,
                role=ConeRole.SKIDPAD_INNER,
            )
        )
        cones.extend(
            self._circle_cones(
                center=right_center,
                radius_m=inner_radius,
                count=inner_count,
                color=ConeColor.YELLOW,
                role=ConeRole.SKIDPAD_INNER,
            )
        )
        cones.extend(
            self._circle_cones(
                center=left_center,
                radius_m=outer_radius,
                count=outer_count,
                color=ConeColor.YELLOW,
                role=ConeRole.SKIDPAD_OUTER,
            )
        )
        cones.extend(
            self._circle_cones(
                center=right_center,
                radius_m=outer_radius,
                count=outer_count,
                color=ConeColor.BLUE,
                role=ConeRole.SKIDPAD_OUTER,
            )
        )
        return cones

    def _circle_cones(
        self,
        center: Point2,
        radius_m: float,
        count: int,
        color: ConeColor,
        role: ConeRole,
    ) -> list[Cone]:
        return [
            Cone(
                x=center[0] + radius_m * math.cos(2.0 * math.pi * i / count),
                y=center[1] + radius_m * math.sin(2.0 * math.pi * i / count),
                color=color,
                size=SMALL_CONE,
                role=role,
            )
            for i in range(count)
        ]

    def _start_finish_cones(self, line: LineMark, normal: Point2) -> list[Cone]:
        tangent = (math.cos(line.heading_rad), math.sin(line.heading_rad))
        side_offset = line.width_m / 2.0 + self.rules.start_finish_cone_offset_m
        cones: list[Cone] = []
        for along in (-self.rules.start_finish_line_guard_m, self.rules.start_finish_line_guard_m):
            for side in (-1.0, 1.0):
                cones.append(
                    Cone(
                        x=line.center[0] + tangent[0] * along + normal[0] * side * side_offset,
                        y=line.center[1] + tangent[1] * along + normal[1] * side * side_offset,
                        color=ConeColor.ORANGE,
                        size=LARGE_CONE,
                        role=ConeRole.START_FINISH,
                        yaw_rad=line.heading_rad,
                    )
                )
        return cones

    def _entry_lane_cones(self, line: LineMark, normal: Point2) -> list[Cone]:
        tangent = (math.cos(line.heading_rad), math.sin(line.heading_rad))
        half_width = line.width_m / 2.0
        cones: list[Cone] = []
        for distance in (-self.rules.autocross_staging_offset_m, -3.0):
            for side in (-1.0, 1.0):
                cones.append(
                    Cone(
                        x=line.center[0] + tangent[0] * distance + normal[0] * side * half_width,
                        y=line.center[1] + tangent[1] * distance + normal[1] * side * half_width,
                        color=ConeColor.ORANGE,
                        size=SMALL_CONE,
                        role=ConeRole.ENTRY_EXIT,
                        yaw_rad=line.heading_rad,
                    )
                )
        return cones

    def _validated_width(self, width_m: float | None) -> float:
        width = self.rules.default_track_width_m if width_m is None else width_m
        if width < self.rules.min_track_width_m:
            raise ValueError(
                f"Track width must be at least {self.rules.min_track_width_m:.1f} m."
            )
        return width

    def _validate_autocross_length(self, length_m: float) -> None:
        if not (
            self.rules.autocross_lap_length_min_m
            <= length_m
            <= self.rules.autocross_lap_length_max_m
        ):
            raise ValueError(
                "Autocross lap length must be "
                f"{self.rules.autocross_lap_length_min_m:.0f}-"
                f"{self.rules.autocross_lap_length_max_m:.0f} m."
            )

    def _cone_spacing(self, cone_density: float) -> float:
        density = float(np.clip(cone_density, 0.0, 1.0))
        return self.rules.default_boundary_cone_spacing_m + (0.5 - density) * 3.0
