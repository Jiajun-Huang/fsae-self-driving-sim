from __future__ import annotations

import argparse
from pathlib import Path

from sim.eval.experiment import run_experiment
from sim.perception.lane_detector import LaneDetector
from sim.planning.waypoint_planner import WaypointPlanner
from sim.roadgen import draw_mesh_debug_in_carla, spawn_cones_in_carla
from sim.roadgen.track_generator import TrackGenerator


def debug_road_preview(
    track_name: str = "debug_track",
    mission: str = "autocross",
    length: float = 260.0,
    width: float = 6.0,
    curvature: float = 0.35,
    cone_density: float = 0.65,
    seed: int = 42,
    output_dir: Path = Path("data/debug"),
) -> None:
    generator = TrackGenerator(seed=seed)
    if mission == "skidpad":
        track = generator.generate_skidpad(
            name=track_name,
            width=width,
            cone_density=cone_density,
        )
    elif mission == "acceleration":
        track = generator.generate_acceleration(
            name=track_name,
            width=width,
            cone_density=cone_density,
        )
    else:
        track = generator.generate_track(
            name=track_name,
            length=length,
            width=width,
            curvature=curvature,
            cone_density=cone_density,
        )
    print(
        f"[debug-road] track={track.name} type={track.track_type} length={track.length:.1f} width={track.width:.1f}"
    )
    print(
        f"[debug-road] waypoints={len(track.waypoints)} cones={len(track.cone_positions)}"
    )
    print(generator.debug_ascii_preview(track, width=60, height=18))

    debug_path = output_dir / f"{track_name}.png"
    generator.debug_plot(track, save_path=debug_path)

    mesh = generator.build_mesh_spec(track)
    obj_path = output_dir / f"{track_name}.obj"
    mesh.export_obj(obj_path)
    print(
        f"[debug-road] mesh_ready vertices={len(mesh.vertices)} triangles={len(mesh.triangles)} "
        f"obj={obj_path}"
    )


def main(load_road_in_carla: bool = False) -> None:
    from sim.env.carla_world import CarlaWorld

    print("[sim] Initializing CARLA simulation scaffold...")

    world = CarlaWorld(host="127.0.0.1", port=2000, timeout=10)
    print("[sim] CARLA world object created.")

    planner = WaypointPlanner(target_speed=8.0)
    detector = LaneDetector()
    track_generator = TrackGenerator(seed=42)

    waypoints = planner.generate_waypoints((0.0, 0.0), (50.0, 0.0), steps=10)
    print("[sim] Waypoints generated:", waypoints[:3], "...")

    track = track_generator.generate_track(
        name="race_track_01",
        length=260.0,
        width=6.0,
        curvature=0.35,
        cone_density=0.65,
    )
    mesh = track_generator.build_mesh_spec(track)
    carla_spec = track_generator.build_carla_import_spec(track, mesh)
    print(
        f"[sim] Generated track: {track.name}, length={track.length:.1f}, "
        f"width={track.width:.1f}, cones={len(track.cone_positions)}"
    )
    print(
        f"[sim] Road mesh spec: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}, "
        f"carla_cones={len(carla_spec.cones)}"
    )
    print("[sim] Sample track waypoint:", track.waypoints[0])
    print("[sim] Sample cone position:", track.cone_positions[0])

    print("[sim] Detector status:", detector.process({"dummy": "data"}))
    print("[sim] Evaluation metrics:", run_experiment())

    if load_road_in_carla:
        world.connect()
        world.set_weather("ClearNoon")
        draw_mesh_debug_in_carla(world, carla_spec)
        cone_actors = spawn_cones_in_carla(world, carla_spec)
        print(f"[sim] Loaded road preview into CARLA: cones={len(cone_actors)}")

    print("[sim] Simulation scaffold ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulation scaffold for CARLA-based perception and planning experiments."
    )
    parser.add_argument(
        "--debug-road",
        action="store_true",
        help="Render a 2D ASCII preview of the abstract race track.",
    )
    parser.add_argument(
        "--mission",
        choices=["autocross", "skidpad", "acceleration"],
        default="autocross",
        help="Driverless mission layout to generate.",
    )
    parser.add_argument(
        "--track-name",
        default="debug_track",
        help="Name for the generated debug track.",
    )
    parser.add_argument(
        "--length", type=float, default=260.0, help="Track length in world units."
    )
    parser.add_argument(
        "--width", type=float, default=6.0, help="Track width in world units."
    )
    parser.add_argument(
        "--curvature", type=float, default=0.35, help="Track curvature factor."
    )
    parser.add_argument(
        "--cone-density", type=float, default=0.65, help="Cone density on the track."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible tracks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/debug"),
        help="Directory for generated PNG and OBJ files, relative to the working directory.",
    )
    parser.add_argument(
        "--skidpad",
        action="store_true",
        help="Compatibility shortcut for --mission skidpad.",
    )
    parser.add_argument(
        "--load-road-in-carla",
        action="store_true",
        help="Connect to CARLA and load the generated road preview.",
    )
    args = parser.parse_args()
    mission = "skidpad" if args.skidpad else args.mission

    if args.debug_road:
        debug_road_preview(
            track_name=args.track_name,
            mission=mission,
            length=args.length,
            width=args.width,
            curvature=args.curvature,
            cone_density=args.cone_density,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        main(load_road_in_carla=args.load_road_in_carla)
