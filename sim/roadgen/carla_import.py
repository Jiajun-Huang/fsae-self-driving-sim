from __future__ import annotations

from dataclasses import dataclass

from .mesh_builder import ConeInstance, MeshSpec
from .models import AbstractTrack


@dataclass(frozen=True)
class CarlaTrackImportSpec:
    mesh: MeshSpec
    cones: list[ConeInstance]
    start_pose: tuple[float, float, float]


def build_carla_import_spec(
    track: AbstractTrack,
    mesh: MeshSpec,
) -> CarlaTrackImportSpec:
    heading = track.start_line.heading_rad if track.start_line else 0.0
    start = track.start_line.center if track.start_line else track.centerline[0]
    return CarlaTrackImportSpec(
        mesh=mesh,
        cones=list(mesh.cone_instances),
        start_pose=(start[0], start[1], heading),
    )


def spawn_cones_in_carla(
    carla_world: object, spec: CarlaTrackImportSpec
) -> list[object]:
    world = getattr(carla_world, "world", carla_world)
    if world is None:
        raise RuntimeError("CARLA world is not connected.")

    import carla

    blueprint_library = world.get_blueprint_library()
    actors: list[object] = []
    for cone in spec.cones:
        try:
            blueprint = blueprint_library.find(cone.blueprint_id)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Cone blueprint '{cone.blueprint_id}' is not registered in CARLA. "
                "Restart CARLA after adding the FASE.Package.json registry file."
            ) from exc
        transform = carla.Transform(
            carla.Location(x=cone.x, y=cone.y, z=cone.z),
            carla.Rotation(yaw=cone.yaw_rad * 57.29577951308232),
        )
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            raise RuntimeError(
                f"Could not spawn FSAE cone '{cone.blueprint_id}' at "
                f"({cone.x:.2f}, {cone.y:.2f}, {cone.z:.2f})."
            )
        actors.append(actor)
    return actors


def draw_mesh_debug_in_carla(
    carla_world: object,
    spec: CarlaTrackImportSpec,
    lifetime: float = 0.0,
) -> None:
    world = getattr(carla_world, "world", carla_world)
    if world is None:
        raise RuntimeError("CARLA world is not connected.")

    import carla

    vertices = spec.mesh.vertices
    for tri in spec.mesh.triangles:
        points = [
            carla.Location(
                x=vertices[idx][0], y=vertices[idx][1], z=vertices[idx][2] + 0.04
            )
            for idx in tri
        ]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            world.debug.draw_line(
                points[a],
                points[b],
                thickness=0.03,
                color=carla.Color(80, 80, 80),
                life_time=lifetime,
            )
