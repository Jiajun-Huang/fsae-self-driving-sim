from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geometry import Point3
from .models import AbstractTrack, Cone

FSAE_CONE_BLUEPRINTS = {
    ("small", "blue"): "static.prop.fsae_small_blue_single_white",
    ("small", "yellow"): "static.prop.fsae_small_yellow_single_black",
    ("small", "orange"): "static.prop.fsae_small_orange_single_white",
    ("large", "orange"): "static.prop.fsae_large_orange_dual_white",
}


@dataclass(frozen=True)
class ConeInstance:
    x: float
    y: float
    z: float
    yaw_rad: float
    color: str
    size: str
    role: str
    blueprint_id: str


@dataclass
class MeshSpec:
    name: str
    vertices: list[Point3]
    triangles: list[tuple[int, int, int]]
    cone_instances: list[ConeInstance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def export_obj(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"o {self.name}"]
        lines.extend(f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in self.vertices)
        for a, b, c in self.triangles:
            lines.append(f"f {a + 1} {b + 1} {c + 1}")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output


class TrackMeshBuilder:
    """Converts an AbstractTrack into road mesh data and CARLA cone instances."""

    def build(self, track: AbstractTrack) -> MeshSpec:
        vertices: list[Point3] = []
        for left, right in zip(track.left_boundary, track.right_boundary):
            vertices.append((left[0], left[1], 0.02))
            vertices.append((right[0], right[1], 0.02))

        triangles: list[tuple[int, int, int]] = []
        segment_count = (
            len(track.centerline) if track.closed else len(track.centerline) - 1
        )
        for i in range(segment_count):
            j = (i + 1) % len(track.centerline)
            left_i = i * 2
            right_i = left_i + 1
            left_j = j * 2
            right_j = left_j + 1
            triangles.append((left_i, right_i, left_j))
            triangles.append((right_i, right_j, left_j))

        return MeshSpec(
            name=track.name,
            vertices=vertices,
            triangles=triangles,
            cone_instances=[self._cone_instance(cone) for cone in track.cones],
            metadata={
                "track_type": track.track_type.value,
                "track_width_m": track.width_m,
                "track_length_m": track.length,
                **track.metadata,
            },
        )

    def _cone_instance(self, cone: Cone) -> ConeInstance:
        key = (cone.size.name.value, cone.color.value)
        try:
            blueprint_id = FSAE_CONE_BLUEPRINTS[key]
        except KeyError as exc:
            raise ValueError(
                f"No FSAE cone blueprint configured for size={key[0]} color={key[1]}."
            ) from exc
        return ConeInstance(
            x=cone.x,
            y=cone.y,
            z=0.0,
            yaw_rad=cone.yaw_rad,
            color=cone.color.value,
            size=cone.size.name.value,
            role=cone.role.value,
            blueprint_id=blueprint_id,
        )
