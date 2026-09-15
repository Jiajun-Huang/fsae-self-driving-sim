from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


SEGMENTS = 64
DOUBLE_SIDED_FACES = True


@dataclass(frozen=True)
class Material:
    name: str
    kd: tuple[float, float, float]
    roughness: float = 0.75

    @property
    def texture_name(self) -> str:
        return f"{self.name}_basecolor.png"


@dataclass(frozen=True)
class ConeSpec:
    slug: str
    label: str
    item_number: str
    body_material: str
    stripe_material: str
    width_cm: float
    height_cm: float
    stripe_ranges_cm: tuple[tuple[float, float], ...]
    mass_kg: float | None


MATERIALS = {
    "traffic_blue": Material("traffic_blue", (0.0, 0.20, 0.55)),
    "traffic_yellow": Material("traffic_yellow", (1.0, 0.78, 0.03)),
    "traffic_orange": Material("traffic_orange", (1.0, 0.28, 0.02)),
    "white_stripe": Material("white_stripe", (0.95, 0.95, 0.90), 0.55),
    "black_stripe": Material("black_stripe", (0.015, 0.015, 0.015), 0.80),
}


SPECS = [
    ConeSpec(
        slug="small_blue_single_white",
        label="Small Blue, Single White Stripe",
        item_number="400.000043.00.00",
        body_material="traffic_blue",
        stripe_material="white_stripe",
        width_cm=22.8,
        height_cm=32.5,
        stripe_ranges_cm=((17.5, 23.0),),
        mass_kg=0.45,
    ),
    ConeSpec(
        slug="small_yellow_single_black",
        label="Small Yellow, Single Black Stripe",
        item_number="400.000013.01.10",
        body_material="traffic_yellow",
        stripe_material="black_stripe",
        width_cm=22.8,
        height_cm=32.5,
        stripe_ranges_cm=((17.5, 23.0),),
        mass_kg=0.45,
    ),
    ConeSpec(
        slug="small_orange_single_white",
        label="Small Orange, Single White Stripe",
        item_number="400.000013.00.00",
        body_material="traffic_orange",
        stripe_material="white_stripe",
        width_cm=22.8,
        height_cm=32.5,
        stripe_ranges_cm=((17.5, 23.0),),
        mass_kg=0.45,
    ),
    ConeSpec(
        slug="large_orange_dual_white",
        label="Large Orange, Dual White Stripes",
        item_number="307.610500.00.00",
        body_material="traffic_orange",
        stripe_material="white_stripe",
        width_cm=28.5,
        height_cm=50.5,
        stripe_ranges_cm=((18.0, 24.0), (31.0, 37.0)),
        mass_kg=1.05,
    ),
]


def material_for_z(spec: ConeSpec, z_cm: float) -> str:
    for low, high in spec.stripe_ranges_cm:
        if low <= z_cm <= high:
            return spec.stripe_material
    return spec.body_material


def write_mtl(path: Path) -> None:
    lines: list[str] = []
    for mat in MATERIALS.values():
        r, g, b = mat.kd
        lines.extend(
            [
                f"newmtl {mat.name}",
                f"Kd {r:.4f} {g:.4f} {b:.4f}",
                "Ka 0.0000 0.0000 0.0000",
                "Ks 0.0800 0.0800 0.0800",
                f"Pr {mat.roughness:.4f}",
                f"map_Kd textures/{mat.texture_name}",
                "illum 2",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_texture_files(output_dir: Path) -> None:
    texture_dir = output_dir / "textures"
    texture_dir.mkdir(exist_ok=True)
    for mat in MATERIALS.values():
        rgb = tuple(round(channel * 255) for channel in mat.kd)
        image = Image.new("RGB", (16, 16), rgb)
        image.save(texture_dir / mat.texture_name)


def add_vertex(
    vertices: list[tuple[float, float, float]],
    uvs: list[tuple[float, float]],
    x: float,
    y: float,
    z: float,
    u: float,
    v: float,
) -> int:
    vertices.append((x, y, z))
    uvs.append((u, v))
    return len(vertices)


def add_face(
    faces: list[tuple[str, tuple[int, ...]]],
    material: str,
    indices: tuple[int, ...],
) -> None:
    faces.append((material, indices))
    if DOUBLE_SIDED_FACES:
        faces.append((material, tuple(reversed(indices))))


def cone_radius_at_z(z: float, body_start: float, body_height: float, bottom_radius: float, top_radius: float) -> float:
    t = max(0.0, min(1.0, (z - body_start) / body_height))
    return bottom_radius + (top_radius - bottom_radius) * t


def generate_cone(spec: ConeSpec, output_dir: Path) -> dict[str, object]:
    base_half = spec.width_cm / 2.0
    base_thickness = 2.5 if spec.height_cm <= 33.0 else 3.5
    collar_height = 1.2 if spec.height_cm <= 33.0 else 1.6
    body_start = base_thickness + collar_height
    body_height = spec.height_cm - body_start
    bottom_radius = spec.width_cm * 0.31
    top_radius = spec.width_cm * 0.055

    z_levels = {body_start, spec.height_cm}
    for low, high in spec.stripe_ranges_cm:
        z_levels.add(low)
        z_levels.add(high)
    for step in range(1, 7):
        z_levels.add(body_start + body_height * step / 7.0)
    z_values = sorted(z for z in z_levels if body_start <= z <= spec.height_cm)

    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[str, tuple[int, ...]]] = []

    base_top_half = base_half - (1.0 if spec.height_cm <= 33.0 else 1.3)
    base = [
        add_vertex(vertices, uvs, -base_half, -base_half, 0.0, 0.0, 0.0),
        add_vertex(vertices, uvs, base_half, -base_half, 0.0, 1.0, 0.0),
        add_vertex(vertices, uvs, base_half, base_half, 0.0, 1.0, 1.0),
        add_vertex(vertices, uvs, -base_half, base_half, 0.0, 0.0, 1.0),
        add_vertex(vertices, uvs, -base_top_half, -base_top_half, base_thickness, 0.0, 0.0),
        add_vertex(vertices, uvs, base_top_half, -base_top_half, base_thickness, 1.0, 0.0),
        add_vertex(vertices, uvs, base_top_half, base_top_half, base_thickness, 1.0, 1.0),
        add_vertex(vertices, uvs, -base_top_half, base_top_half, base_thickness, 0.0, 1.0),
    ]
    body_mat = spec.body_material
    add_face(faces, body_mat, (base[0], base[3], base[2], base[1]))
    add_face(faces, body_mat, (base[4], base[5], base[6], base[7]))
    add_face(faces, body_mat, (base[0], base[1], base[5], base[4]))
    add_face(faces, body_mat, (base[1], base[2], base[6], base[5]))
    add_face(faces, body_mat, (base[2], base[3], base[7], base[6]))
    add_face(faces, body_mat, (base[3], base[0], base[4], base[7]))

    collar_outer_radius = min(base_top_half * 0.86, bottom_radius * 1.28)
    collar_lower: list[int] = []
    collar_upper: list[int] = []
    for i in range(SEGMENTS):
        angle = 2.0 * math.pi * i / SEGMENTS
        collar_lower.append(
            add_vertex(
                vertices,
                uvs,
                collar_outer_radius * math.cos(angle),
                collar_outer_radius * math.sin(angle),
                base_thickness,
                i / SEGMENTS,
                base_thickness / spec.height_cm,
            )
        )
        collar_upper.append(
            add_vertex(
                vertices,
                uvs,
                bottom_radius * math.cos(angle),
                bottom_radius * math.sin(angle),
                body_start,
                i / SEGMENTS,
                body_start / spec.height_cm,
            )
        )
    for i in range(SEGMENTS):
        j = (i + 1) % SEGMENTS
        add_face(faces, body_mat, (collar_lower[i], collar_lower[j], collar_upper[j], collar_upper[i]))

    rings: list[list[int]] = []
    for z in z_values:
        radius = cone_radius_at_z(z, body_start, body_height, bottom_radius, top_radius)
        ring: list[int] = []
        for i in range(SEGMENTS):
            angle = 2.0 * math.pi * i / SEGMENTS
            ring.append(
                add_vertex(
                    vertices,
                    uvs,
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    z,
                    i / SEGMENTS,
                    z / spec.height_cm,
                )
            )
        rings.append(ring)

    for lower_idx in range(len(rings) - 1):
        lower = rings[lower_idx]
        upper = rings[lower_idx + 1]
        z_mid = (z_values[lower_idx] + z_values[lower_idx + 1]) / 2.0
        mat = material_for_z(spec, z_mid)
        for i in range(SEGMENTS):
            j = (i + 1) % SEGMENTS
            add_face(faces, mat, (lower[i], lower[j], upper[j], upper[i]))

    bottom_center = add_vertex(vertices, uvs, 0.0, 0.0, body_start, 0.5, 0.5)
    for i in range(SEGMENTS):
        j = (i + 1) % SEGMENTS
        add_face(faces, body_mat, (bottom_center, rings[0][i], rings[0][j]))

    top_center = add_vertex(vertices, uvs, 0.0, 0.0, spec.height_cm, 0.5, 0.5)
    top_mat = material_for_z(spec, spec.height_cm)
    for i in range(SEGMENTS):
        j = (i + 1) % SEGMENTS
        add_face(faces, top_mat, (top_center, rings[-1][i], rings[-1][j]))

    obj_path = output_dir / f"fsae_{spec.slug}.obj"
    mtl_path = output_dir / "fsae_cones.mtl"
    lines = [
        f"# {spec.label}",
        f"# WEMAS {spec.item_number}",
        "# Units: centimeters. Unreal Engine uses centimeters by default.",
        f"mtllib {mtl_path.name}",
        "",
    ]
    current_material = ""
    for x, y, z in vertices:
        lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
    lines.append("")
    for u, v in uvs:
        lines.append(f"vt {u:.5f} {v:.5f}")
    lines.append("")
    for material, indices in faces:
        if material != current_material:
            lines.append(f"usemtl {material}")
            current_material = material
        lines.append("f " + " ".join(f"{index}/{index}" for index in indices))
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "slug": spec.slug,
        "label": spec.label,
        "wemas_part_number": spec.item_number,
        "obj": obj_path.name,
        "mtl": mtl_path.name,
        "dimensions_mm": {
            "length": round(spec.width_cm * 10),
            "width": round(spec.width_cm * 10),
            "height": round(spec.height_cm * 10),
        },
        "mass_kg": spec.mass_kg,
        "body_material": spec.body_material,
        "stripe_material": spec.stripe_material,
        "stripe_count": len(spec.stripe_ranges_cm),
    }


def write_readme(output_dir: Path, metadata: list[dict[str, object]]) -> None:
    table = "\n".join(
        "| {label} | {part} | {size} | {obj} |".format(
            label=item["label"],
            part=item["wemas_part_number"],
            size="{length} x {width} x {height} mm".format(**item["dimensions_mm"]),
            obj=item["obj"],
        )
        for item in metadata
    )
    text = f"""# FSAE Cone Models

These OBJ models are generated for the Formula SAE Driverless cone set.

OBJ units are centimeters, matching Unreal Engine's default world units.
The MTL file references solid-color PNG textures in `textures/` so UE4 imports visible colored materials instead of default gray slots.

| Cone | WEMAS part number | Dimensions | File |
| --- | --- | --- | --- |
{table}

## CARLA / UE4 Import

The UE4 4.26.2 assets and prop registry are in `carla/FASE/`.
Copy their contents to `carla/Unreal/CarlaUE4/Content/Carla/FASE/` with CARLA closed,
then restart it. Keep the name `FASE` because the registry and asset references
use `/Game/Carla/FASE/`.

For a different engine version, import the OBJ files and keep the MTL and
`textures/` directory alongside them. Save the meshes with the names in
`carla/FASE/FASE.Package.json` and configure collision for LiDAR visibility.
See [CARLA Windows Setup](../../../CARLA_WINDOWS_SETUP.md) for full instructions.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    write_texture_files(output_dir)
    write_mtl(output_dir / "fsae_cones.mtl")
    metadata = [generate_cone(spec, output_dir) for spec in SPECS]
    (output_dir / "fsae_cones.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, metadata)
    for item in metadata:
        print(f"generated {item['obj']}")


if __name__ == "__main__":
    main()
