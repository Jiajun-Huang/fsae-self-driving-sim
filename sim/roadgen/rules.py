from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackType(str, Enum):
    ACCELERATION = "acceleration"
    SKIDPAD = "skidpad"
    AUTOCROSS = "autocross"


class ConeColor(str, Enum):
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"


class ConeSizeName(str, Enum):
    SMALL = "small"
    LARGE = "large"


class ConeRole(str, Enum):
    LEFT_BOUNDARY = "left_boundary"
    RIGHT_BOUNDARY = "right_boundary"
    ENTRY_EXIT = "entry_exit"
    START_FINISH = "start_finish"
    SKIDPAD_INNER = "skidpad_inner"
    SKIDPAD_OUTER = "skidpad_outer"


@dataclass(frozen=True)
class ConeSize:
    name: ConeSizeName
    base_x_m: float
    base_y_m: float
    height_m: float


@dataclass(frozen=True)
class FSAERuleProfile:
    min_track_width_m: float = 3.0
    min_turning_diameter_m: float = 9.0
    max_autocross_straight_m: float = 80.0
    autocross_lap_length_min_m: float = 200.0
    autocross_lap_length_max_m: float = 500.0
    acceleration_course_length_m: float = 75.0
    acceleration_stop_length_m: float = 75.0
    autocross_stop_length_m: float = 30.0
    skidpad_stop_length_m: float = 25.0
    autocross_staging_offset_m: float = 6.0
    acceleration_staging_offset_m: float = 0.30
    skidpad_inner_cones_per_circle: int = 17
    skidpad_inner_radius_m: float = 7.625
    skidpad_outer_radius_m: float = 10.625
    skidpad_center_distance_m: float = 18.25
    default_track_width_m: float = 6.0
    default_boundary_cone_spacing_m: float = 5.0
    start_finish_cone_offset_m: float = 0.8
    start_finish_line_guard_m: float = 1.0


SMALL_CONE = ConeSize(
    name=ConeSizeName.SMALL,
    base_x_m=0.228,
    base_y_m=0.228,
    height_m=0.325,
)

LARGE_CONE = ConeSize(
    name=ConeSizeName.LARGE,
    base_x_m=0.285,
    base_y_m=0.285,
    height_m=0.505,
)

DEFAULT_RULES = FSAERuleProfile()

