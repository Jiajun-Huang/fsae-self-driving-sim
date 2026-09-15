from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geometry import Point2, polyline_length
from .rules import ConeColor, ConeRole, ConeSize, TrackType


@dataclass(frozen=True)
class LineMark:
    center: Point2
    heading_rad: float
    width_m: float
    kind: str


@dataclass(frozen=True)
class Cone:
    x: float
    y: float
    color: ConeColor
    size: ConeSize
    role: ConeRole
    yaw_rad: float = 0.0

    @property
    def position(self) -> Point2:
        return (self.x, self.y)


@dataclass
class AbstractTrack:
    name: str
    track_type: TrackType
    centerline: list[Point2]
    left_boundary: list[Point2]
    right_boundary: list[Point2]
    cones: list[Cone]
    width_m: float
    closed: bool
    start_line: LineMark | None = None
    finish_line: LineMark | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return polyline_length(self.centerline, closed=self.closed)

    @property
    def width(self) -> float:
        return self.width_m

    @property
    def waypoints(self) -> list[Point2]:
        return list(self.centerline)

    @property
    def cone_positions(self) -> list[tuple[float, float, str]]:
        return [(cone.x, cone.y, cone.color.value) for cone in self.cones]

    def cones_by_role(self, role: ConeRole) -> list[Cone]:
        return [cone for cone in self.cones if cone.role == role]

