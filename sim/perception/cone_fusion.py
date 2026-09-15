from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CameraConeDetection:
    color: str
    x_m: float
    y_m: float
    confidence: float = 1.0


@dataclass(frozen=True)
class LidarConeDetection:
    x_m: float
    y_m: float
    confidence: float = 1.0


@dataclass(frozen=True)
class FusedCone:
    color: str
    x_m: float
    y_m: float
    confidence: float
    source: str


def distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CameraLidarConeFusion:
    def __init__(
        self,
        association_radius_m: float = 0.8,
        lidar_position_weight: float = 0.75,
    ) -> None:
        self.association_radius_m = association_radius_m
        self.lidar_position_weight = lidar_position_weight

    def fuse(
        self,
        camera_detections: Iterable[CameraConeDetection],
        lidar_detections: Iterable[LidarConeDetection],
    ) -> list[FusedCone]:
        camera = list(camera_detections)
        lidar = list(lidar_detections)
        matched_lidar: set[int] = set()
        fused: list[FusedCone] = []

        for cam in camera:
            match_index = self._nearest_lidar(cam, lidar, matched_lidar)
            if match_index is None:
                # Camera-only cones keep color but are less trusted because depth is weaker.
                fused.append(
                    FusedCone(
                        color=cam.color,
                        x_m=cam.x_m,
                        y_m=cam.y_m,
                        confidence=0.55 * cam.confidence,
                        source="camera",
                    )
                )
                continue

            lid = lidar[match_index]
            matched_lidar.add(match_index)
            lidar_w = self.lidar_position_weight
            camera_w = 1.0 - lidar_w
            fused.append(
                FusedCone(
                    color=cam.color,
                    x_m=lid.x_m * lidar_w + cam.x_m * camera_w,
                    y_m=lid.y_m * lidar_w + cam.y_m * camera_w,
                    confidence=min(1.0, 0.5 * cam.confidence + 0.5 * lid.confidence),
                    source="camera_lidar",
                )
            )

        for index, lid in enumerate(lidar):
            if index in matched_lidar:
                continue
            # LiDAR-only cones have reliable position but unknown color, so lane search treats them as neutral.
            fused.append(
                FusedCone(
                    color="unknown",
                    x_m=lid.x_m,
                    y_m=lid.y_m,
                    confidence=0.65 * lid.confidence,
                    source="lidar",
                )
            )

        return fused

    def _nearest_lidar(
        self,
        camera_detection: CameraConeDetection,
        lidar_detections: list[LidarConeDetection],
        matched_lidar: set[int],
    ) -> int | None:
        camera_xy = (camera_detection.x_m, camera_detection.y_m)
        candidates: list[tuple[float, int]] = []
        for index, lidar_detection in enumerate(lidar_detections):
            if index in matched_lidar:
                continue
            lidar_xy = (lidar_detection.x_m, lidar_detection.y_m)
            distance = distance_2d(camera_xy, lidar_xy)
            if distance <= self.association_radius_m:
                candidates.append((distance, index))
        return min(candidates)[1] if candidates else None
