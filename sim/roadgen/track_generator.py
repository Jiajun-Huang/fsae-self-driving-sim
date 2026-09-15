from __future__ import annotations

from pathlib import Path

from .abstract_generator import AbstractTrackGenerator
from .carla_import import CarlaTrackImportSpec, build_carla_import_spec
from .debug import ascii_preview, plot_debug_track
from .mesh_builder import MeshSpec, TrackMeshBuilder
from .models import AbstractTrack
from .rules import DEFAULT_RULES, FSAERuleProfile


class TrackGenerator:
    """Facade that keeps the two road generation stages easy to call."""

    def __init__(
        self,
        seed: int | None = None,
        rules: FSAERuleProfile = DEFAULT_RULES,
    ) -> None:
        self.rules = rules
        self.abstract = AbstractTrackGenerator(seed=seed, rules=rules)
        self.mesh_builder = TrackMeshBuilder()

    def generate_track(
        self,
        name: str = "autocross",
        length: float = 260.0,
        width: float | None = None,
        curvature: float = 0.35,
        cone_density: float = 0.65,
    ) -> AbstractTrack:
        return self.generate_autocross(
            name=name,
            length=length,
            width=width,
            curvature=curvature,
            cone_density=cone_density,
        )

    def generate_autocross(
        self,
        name: str = "autocross",
        length: float = 260.0,
        width: float | None = None,
        curvature: float = 0.35,
        cone_density: float = 0.65,
    ) -> AbstractTrack:
        return self.abstract.generate_autocross(
            name=name,
            length_m=length,
            width_m=width,
            curvature=curvature,
            cone_density=cone_density,
        )

    def generate_skidpad(
        self,
        name: str = "skidpad",
        length: float | None = None,
        width: float | None = None,
        cone_density: float = 0.85,
    ) -> AbstractTrack:
        del length
        return self.abstract.generate_skidpad(
            name=name,
            width_m=width,
            cone_density=cone_density,
        )

    def generate_acceleration(
        self,
        name: str = "acceleration",
        width: float | None = None,
        cone_density: float = 0.65,
    ) -> AbstractTrack:
        return self.abstract.generate_acceleration(
            name=name,
            width_m=width,
            cone_density=cone_density,
        )

    def build_mesh_spec(self, track: AbstractTrack) -> MeshSpec:
        return self.mesh_builder.build(track)

    def build_carla_import_spec(
        self,
        track: AbstractTrack,
        mesh: MeshSpec | None = None,
    ) -> CarlaTrackImportSpec:
        mesh_spec = self.build_mesh_spec(track) if mesh is None else mesh
        return build_carla_import_spec(track, mesh_spec)

    def debug_plot(self, track: AbstractTrack, save_path: str | Path) -> Path:
        return plot_debug_track(track, save_path)

    def debug_ascii_preview(
        self,
        track: AbstractTrack,
        width: int = 72,
        height: int = 24,
    ) -> str:
        return ascii_preview(track, width=width, height=height)
