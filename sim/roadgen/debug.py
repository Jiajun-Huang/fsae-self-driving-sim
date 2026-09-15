from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import AbstractTrack


PLOT_COLORS = {
    "blue": "#2563eb",
    "yellow": "#eab308",
    "orange": "#f97316",
}


def plot_debug_track(track: AbstractTrack, save_path: str | Path) -> Path:
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    left = np.asarray(track.left_boundary)
    right = np.asarray(track.right_boundary)
    center = np.asarray(track.centerline)

    ax.plot(left[:, 0], left[:, 1], color="#111827", linewidth=1.2)
    ax.plot(right[:, 0], right[:, 1], color="#111827", linewidth=1.2)
    ax.plot(center[:, 0], center[:, 1], color="#6b7280", linewidth=0.8, linestyle="--")

    for color, marker_color in PLOT_COLORS.items():
        cones = [cone for cone in track.cones if cone.color.value == color]
        if not cones:
            continue
        size = [36 if cone.size.name.value == "large" else 16 for cone in cones]
        xs = [cone.x for cone in cones]
        ys = [cone.y for cone in cones]
        ax.scatter(xs, ys, s=size, c=marker_color, edgecolors="#111827", linewidths=0.3)

    if track.start_line is not None:
        ax.scatter(
            [track.start_line.center[0]],
            [track.start_line.center[1]],
            s=42,
            c="#10b981",
            marker="x",
            linewidths=1.5,
        )

    ax.set_title(f"{track.name} ({track.track_type.value})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.2, color="#d1d5db")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def ascii_preview(track: AbstractTrack, width: int = 72, height: int = 24) -> str:
    points = list(track.centerline)
    points.extend((cone.x, cone.y) for cone in track.cones)
    arr = np.asarray(points, dtype=float)
    min_xy = arr.min(axis=0)
    max_xy = arr.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1.0)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def put(x: float, y: float, char: str) -> None:
        norm = (np.array([x, y]) - min_xy) / span
        gx = int(np.clip(round(norm[0] * (width - 1)), 0, width - 1))
        gy = int(np.clip(round((1.0 - norm[1]) * (height - 1)), 0, height - 1))
        grid[gy][gx] = char

    for x, y in track.centerline:
        put(x, y, ".")
    for cone in track.cones:
        char = {"blue": "B", "yellow": "Y", "orange": "O"}[cone.color.value]
        put(cone.x, cone.y, char)
    if track.start_line is not None:
        put(track.start_line.center[0], track.start_line.center[1], "S")

    return "\n".join("".join(row).rstrip() for row in grid)

