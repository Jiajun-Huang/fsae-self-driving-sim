from __future__ import annotations

import math
from typing import Iterable

import numpy as np

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def as_points(points: Iterable[Point2]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected an iterable of 2D points.")
    if len(arr) < 2:
        raise ValueError("At least two points are required.")
    return arr


def polyline_length(points: Iterable[Point2], closed: bool = False) -> float:
    arr = as_points(points)
    if closed:
        arr = np.vstack([arr, arr[0]])
    deltas = arr[1:] - arr[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def resample_polyline(
    points: Iterable[Point2],
    spacing_m: float,
    closed: bool = False,
) -> list[Point2]:
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive.")

    arr = as_points(points)
    if closed:
        arr = np.vstack([arr, arr[0]])

    deltas = arr[1:] - arr[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total == 0.0:
        raise ValueError("Cannot resample a zero-length polyline.")

    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.arange(0.0, total, spacing_m)
    if not closed and (len(samples) == 0 or not math.isclose(samples[-1], total)):
        samples = np.append(samples, total)

    result: list[Point2] = []
    for dist in samples:
        if dist >= total:
            result.append((float(arr[-1, 0]), float(arr[-1, 1])))
            continue

        idx = int(np.searchsorted(cumulative, dist, side="right") - 1)
        idx = min(max(idx, 0), len(lengths) - 1)
        seg_len = lengths[idx]
        ratio = 0.0 if seg_len == 0.0 else (dist - cumulative[idx]) / seg_len
        point = arr[idx] + ratio * deltas[idx]
        result.append((float(point[0]), float(point[1])))

    return result


def headings_and_normals(
    points: Iterable[Point2],
    closed: bool,
) -> tuple[list[float], list[Point2]]:
    arr = as_points(points)
    if closed:
        tangents = np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)
    else:
        tangents = np.zeros_like(arr)
        tangents[0] = arr[1] - arr[0]
        tangents[-1] = arr[-1] - arr[-2]
        if len(arr) > 2:
            tangents[1:-1] = arr[2:] - arr[:-2]

    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0.0] = 1.0
    tangents = tangents / norms[:, None]
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])

    return (
        [float(h) for h in headings],
        [(float(n[0]), float(n[1])) for n in normals],
    )


def offset_polyline(
    points: Iterable[Point2],
    width_m: float,
    closed: bool,
) -> tuple[list[Point2], list[Point2]]:
    arr = as_points(points)
    _, normals = headings_and_normals(points=arr, closed=closed)
    normal_arr = np.asarray(normals, dtype=float)
    half_width = width_m / 2.0
    left = arr + normal_arr * half_width
    right = arr - normal_arr * half_width
    return (
        [(float(p[0]), float(p[1])) for p in left],
        [(float(p[0]), float(p[1])) for p in right],
    )


def catmull_rom_closed(
    control_points: Iterable[Point2],
    samples_per_segment: int = 16,
) -> list[Point2]:
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")

    pts = as_points(control_points)
    result: list[Point2] = []
    for i in range(len(pts)):
        p0 = pts[(i - 1) % len(pts)]
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        p3 = pts[(i + 2) % len(pts)]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            result.append((float(point[0]), float(point[1])))
    return result


def scale_points(points: Iterable[Point2], factor: float) -> list[Point2]:
    arr = as_points(points) * factor
    return [(float(p[0]), float(p[1])) for p in arr]


def center_points(points: Iterable[Point2]) -> list[Point2]:
    arr = as_points(points)
    centroid = arr.mean(axis=0)
    arr = arr - centroid
    return [(float(p[0]), float(p[1])) for p in arr]


def point_at_distance(
    points: Iterable[Point2],
    distance_m: float,
    closed: bool,
) -> Point2:
    arr = as_points(points)
    if closed:
        arr = np.vstack([arr, arr[0]])

    deltas = arr[1:] - arr[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    total = float(lengths.sum())
    if total == 0.0:
        raise ValueError("Cannot sample a zero-length polyline.")

    dist = distance_m % total if closed else min(max(distance_m, 0.0), total)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    idx = int(np.searchsorted(cumulative, dist, side="right") - 1)
    idx = min(max(idx, 0), len(lengths) - 1)
    seg_len = lengths[idx]
    ratio = 0.0 if seg_len == 0.0 else (dist - cumulative[idx]) / seg_len
    point = arr[idx] + ratio * deltas[idx]
    return (float(point[0]), float(point[1]))

