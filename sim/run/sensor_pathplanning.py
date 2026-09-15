from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

import cv2

import carla
from sim.planning.paper_stack_planner import ConeLandmark, DelaunayMinimumCurvaturePlanner
from sim.roadgen import spawn_cones_in_carla
from sim.roadgen.geometry import Point2
from sim.roadgen.track_generator import TrackGenerator
from sim.run.pathplanning import (
    MAX_ACCEL_MPS2,
    MAX_BRAKE_MPS2,
    MAX_LATERAL_ACCEL_MPS2,
    MAX_SPEED_LOOKAHEAD_M,
    MAX_SPEED_MPS,
    MIN_SPEED_LOOKAHEAD_M,
    MIN_SPEED_MPS,
    LateralMPCController,
    LongitudinalPID,
    build_reference_trajectory,
    clamp,
    draw_reference_trajectory,
    format_speed,
    nearest_reference_index,
    read_vehicle_state,
    setup_client,
    setup_world,
    signed_lateral_error,
    spawn_car,
    target_speed_ahead,
)


CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAMERA_FOV_DEG = 90.0
LIDAR_RANGE_M = 45.0
LIDAR_PROPOSAL_RANGE_M = 35.0
LIDAR_CLUSTER_RADIUS_M = 0.38
LIDAR_MIN_CLUSTER_POINTS = 2
MAP_ASSOCIATION_RADIUS_M = 0.7
MAP_STABLE_CONFIDENCE = 2.0
MIN_LANDMARKS_FOR_PLANNING = 8
MIN_PATH_POINTS_FOR_CONTROL = 5
CAMERA_WINDOW_NAME = "FSAE camera"
LIDAR_WINDOW_NAME = "FSAE lidar top-down"
LIDAR_VIEW_SIZE_PX = 640
LIDAR_VIEW_RADIUS_M = 40.0


@dataclass(frozen=True)
class SensorFrame:
    # Step 1: keep the latest raw RGB image and its CARLA world transform together.
    rgb_image: np.ndarray | None
    camera_transform: carla.Transform | None

    # Step 2: keep the latest LiDAR point cloud in sensor coordinates.
    lidar_points: np.ndarray | None
    lidar_transform: carla.Transform | None


@dataclass(frozen=True)
class LidarCluster:
    # Step 1: one cluster is one possible physical cone position.
    x: float
    y: float
    z: float
    confidence: float


class LatestSensorBuffer:
    """Thread-safe cache for CARLA asynchronous sensor callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rgb_image: np.ndarray | None = None
        self._camera_transform: carla.Transform | None = None
        self._lidar_points: np.ndarray | None = None
        self._lidar_transform: carla.Transform | None = None

    def update_camera(self, image: carla.Image) -> None:
        # Step 1: CARLA image bytes are BGRA, so reshape them first.
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )

        # Step 2: convert BGRA to RGB because color rules are easier to read in RGB.
        rgb = bgra[:, :, [2, 1, 0]].copy()

        # Step 3: publish the latest camera frame atomically.
        with self._lock:
            self._rgb_image = rgb
            self._camera_transform = image.transform

    def update_lidar(self, cloud: carla.LidarMeasurement) -> None:
        # Step 1: CARLA LiDAR stores x, y, z, intensity as float32 tuples.
        points = np.frombuffer(cloud.raw_data, dtype=np.float32).reshape((-1, 4))

        # Step 2: copy the point array because the callback buffer is reused by CARLA.
        with self._lock:
            self._lidar_points = points.copy()
            self._lidar_transform = cloud.transform

    def snapshot(self) -> SensorFrame:
        # Step 1: copy references under one lock so camera/LiDAR data are consistent enough.
        with self._lock:
            return SensorFrame(
                rgb_image=None if self._rgb_image is None else self._rgb_image.copy(),
                camera_transform=self._camera_transform,
                lidar_points=None
                if self._lidar_points is None
                else self._lidar_points.copy(),
                lidar_transform=self._lidar_transform,
            )


class CameraLidarConeDetector:
    """Convert CARLA camera and LiDAR packets into colored cone landmarks."""

    def __init__(self) -> None:
        self.last_cluster_count = 0
        self.last_unknown_color_count = 0
        self.last_candidate_count = 0
        self.last_ground_z = 0.0
        self.last_clusters: list[LidarCluster] = []
        self.last_projected_cluster_count = 0
        self.last_colored_patch_count = 0

    def detect(self, frame: SensorFrame) -> list[ConeLandmark]:
        # Step 1: without a LiDAR packet, there is no metric cone position.
        self.last_cluster_count = 0
        self.last_unknown_color_count = 0
        self.last_candidate_count = 0
        self.last_ground_z = 0.0
        self.last_clusters = []
        self.last_projected_cluster_count = 0
        self.last_colored_patch_count = 0
        if frame.lidar_points is None or frame.lidar_transform is None:
            return []

        # Step 2: cluster LiDAR hits into cone-sized objects in world coordinates.
        clusters = self._lidar_clusters(frame.lidar_points, frame.lidar_transform)
        self.last_cluster_count = len(clusters)
        self.last_clusters = clusters

        # Step 3: use the camera image to assign each LiDAR cluster a cone color.
        landmarks: list[ConeLandmark] = []
        for cluster in clusters:
            color = self._classify_cluster_color(cluster, frame)
            if color == "unknown":
                self.last_unknown_color_count += 1
                continue
            landmarks.append(
                ConeLandmark(
                    x=cluster.x,
                    y=cluster.y,
                    color=color,
                    confidence=cluster.confidence,
                )
            )
        return landmarks

    def _lidar_clusters(
        self,
        sensor_points: np.ndarray,
        lidar_transform: carla.Transform,
    ) -> list[LidarCluster]:
        # Step 1: keep finite points within the LiDAR proposal range.
        finite = np.isfinite(sensor_points[:, :3]).all(axis=1)
        range_xy = np.linalg.norm(sensor_points[:, :2], axis=1) <= LIDAR_PROPOSAL_RANGE_M
        local_points = sensor_points[finite & range_xy]
        if len(local_points) == 0:
            return []

        # Step 2: follow the paper-style LiDAR front-end: remove the ground first,
        # then cluster the remaining cone-sized objects as landmark proposals.
        ground_z = float(np.percentile(local_points[:, 2], 5.0))
        self.last_ground_z = ground_z
        height_above_ground = local_points[:, 2] - ground_z
        cone_height = (height_above_ground > 0.20) & (height_above_ground < 1.45)
        local_cone_points = local_points[cone_height, :3]
        self.last_candidate_count = len(local_cone_points)
        if len(local_cone_points) == 0:
            return []

        # Step 3: transform cone candidates into CARLA world coordinates.
        cone_points = transform_points(local_cone_points, lidar_transform)

        # Step 4: group nearby points by Euclidean distance in the ground plane.
        clusters = euclidean_clusters_xy(
            cone_points,
            radius_m=LIDAR_CLUSTER_RADIUS_M,
            min_points=LIDAR_MIN_CLUSTER_POINTS,
        )

        # Step 5: reject clusters that are too large to be a cone.
        result: list[LidarCluster] = []
        for cluster in clusters:
            extent = cluster.max(axis=0) - cluster.min(axis=0)
            if extent[0] > 0.90 or extent[1] > 0.90 or extent[2] > 1.35:
                continue
            center = cluster.mean(axis=0)
            confidence = clamp(len(cluster) / 18.0, 0.25, 1.0)
            result.append(
                LidarCluster(
                    x=float(center[0]),
                    y=float(center[1]),
                    z=float(center[2]),
                    confidence=confidence,
                )
            )
        return result

    def _classify_cluster_color(
        self,
        cluster: LidarCluster,
        frame: SensorFrame,
    ) -> str:
        # Step 1: a camera frame is required for color; LiDAR alone gives position only.
        if frame.rgb_image is None or frame.camera_transform is None:
            return "unknown"

        # Step 2: project several heights because a sparse LiDAR cluster may sit on
        # the cone side, base, or tip rather than exactly at the visual center.
        best_color = "unknown"
        best_count = 0
        best_required = 1
        projected = False
        for z_offset in (0.05, 0.25, 0.45, 0.70):
            pixel = project_world_to_camera(
                point=(cluster.x, cluster.y, cluster.z + z_offset),
                camera_transform=frame.camera_transform,
                width=frame.rgb_image.shape[1],
                height=frame.rgb_image.shape[0],
                fov_deg=CAMERA_FOV_DEG,
            )
            if pixel is None:
                continue
            if pixel[1] < int(frame.rgb_image.shape[0] * 0.30):
                continue
            projected = True

            # Step 3: sample a generous patch around the projected point.
            u, v = pixel
            patch_radius = 28
            x0 = max(u - patch_radius, 0)
            x1 = min(u + patch_radius + 1, frame.rgb_image.shape[1])
            y0 = max(v - patch_radius, 0)
            y1 = min(v + patch_radius + 1, frame.rgb_image.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue

            # Step 4: keep the color with the strongest HSV evidence.
            patch = frame.rgb_image[y0:y1, x0:x1]
            color, count, required = score_fsae_cone_patch(patch)
            if count > best_count:
                best_color = color
                best_count = count
                best_required = required

        if projected:
            self.last_projected_cluster_count += 1
        if best_color != "unknown" and best_count >= best_required * 2:
            self.last_colored_patch_count += 1
            return best_color
        return "unknown"


class OnlineConeMap:
    """Merge repeated sensor detections into a small global cone map."""

    def __init__(
        self,
        association_radius_m: float = MAP_ASSOCIATION_RADIUS_M,
        stable_confidence: float = MAP_STABLE_CONFIDENCE,
    ) -> None:
        self.association_radius_m = association_radius_m
        self.stable_confidence = stable_confidence
        self._landmarks: list[ConeLandmark] = []

    @property
    def landmarks(self) -> list[ConeLandmark]:
        # Step 1: expose only landmarks seen enough times to suppress one-frame false positives.
        return [
            landmark
            for landmark in self._landmarks
            if landmark.confidence >= self.stable_confidence
        ]

    @property
    def raw_landmarks(self) -> list[ConeLandmark]:
        # Step 1: raw landmarks are useful for debugging proposal accumulation.
        return list(self._landmarks)

    def update(self, detections: Sequence[ConeLandmark]) -> list[ConeLandmark]:
        # Step 1: insert new cones or confidence-average them into existing cones.
        for detection in detections:
            index = self._nearest_same_color(detection)
            if index is None:
                self._landmarks.append(detection)
                continue

            old = self._landmarks[index]
            total_confidence = max(old.confidence + detection.confidence, 1e-6)
            self._landmarks[index] = ConeLandmark(
                x=(old.x * old.confidence + detection.x * detection.confidence)
                / total_confidence,
                y=(old.y * old.confidence + detection.y * detection.confidence)
                / total_confidence,
                color=old.color,
                confidence=clamp(total_confidence, 0.1, 10.0),
            )
        return self.landmarks

    def _nearest_same_color(self, detection: ConeLandmark) -> int | None:
        # Step 1: only merge cones with the same camera-observed color.
        best_index: int | None = None
        best_distance = self.association_radius_m
        for index, landmark in enumerate(self._landmarks):
            if landmark.color != detection.color:
                continue
            distance = math.hypot(landmark.x - detection.x, landmark.y - detection.y)
            if distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index


def require_opencv_imshow() -> None:
    # Step 1: create a tiny image to verify that HighGUI can really open windows.
    test_window = "__opencv_imshow_test__"
    test_image = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.imshow(test_window, test_image)
        cv2.waitKey(1)
        cv2.destroyWindow(test_window)
    except cv2.error as exc:
        # Step 2: stop here; this is an environment problem, not a perception fallback.
        raise RuntimeError(
            "Your current cv2 package was built without a working window backend, "
            "so cv2.imshow cannot be used. Fix the same Python environment with:\n\n"
            "python -m pip uninstall -y opencv-python-headless "
            "opencv-contrib-python-headless opencv-python opencv-contrib-python\n"
            "python -m pip install opencv-python==4.10.0.84\n\n"
            "Then verify it with:\n"
            "python -c \"import cv2, numpy as np; "
            "img=np.zeros((120,160,3),np.uint8); "
            "cv2.imshow('opencv test', img); cv2.waitKey(1000); "
            "cv2.destroyAllWindows()\""
        ) from exc


class SensorDebugViewer:
    """OpenCV windows for live camera and LiDAR debugging."""

    def __init__(self) -> None:
        # Step 1: fail immediately if this Python has a headless OpenCV build.
        require_opencv_imshow()

    def show(
        self,
        frame: SensorFrame,
        clusters: Sequence[LidarCluster],
        detections: Sequence[ConeLandmark],
        landmarks: Sequence[ConeLandmark],
    ) -> None:
        # Step 1: render both windows from the same sensor snapshot.
        camera_view = self._camera_view(frame, clusters, detections)
        lidar_view = self._lidar_view(frame, clusters, detections, landmarks)
        cv2.imshow(CAMERA_WINDOW_NAME, camera_view)
        cv2.imshow(LIDAR_WINDOW_NAME, lidar_view)

        # Step 2: pump the GUI event loop; pressing q closes the debug windows.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.close()

    def close(self) -> None:
        # Step 1: close OpenCV windows without touching CARLA actors.
        try:
            cv2.destroyWindow(CAMERA_WINDOW_NAME)
            cv2.destroyWindow(LIDAR_WINDOW_NAME)
        except cv2.error:
            pass

    def _camera_view(
        self,
        frame: SensorFrame,
        clusters: Sequence[LidarCluster],
        detections: Sequence[ConeLandmark],
    ) -> np.ndarray:
        # Step 1: show a black placeholder until the first camera frame arrives.
        if frame.rgb_image is None:
            image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                image,
                "waiting for camera",
                (32, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            return image

        # Step 2: OpenCV expects BGR even though the detector uses RGB.
        image = frame.rgb_image[:, :, ::-1].copy()

        # Step 3: draw LiDAR clusters before color classification in magenta.
        if frame.camera_transform is not None:
            for cluster in clusters:
                pixel = project_world_to_camera(
                    point=(cluster.x, cluster.y, cluster.z + 0.20),
                    camera_transform=frame.camera_transform,
                    width=image.shape[1],
                    height=image.shape[0],
                    fov_deg=CAMERA_FOV_DEG,
                )
                if pixel is not None:
                    cv2.circle(image, pixel, 10, (255, 0, 255), 2, cv2.LINE_AA)

            # Step 4: draw each current colored cone detection on top of clusters.
            for detection in detections:
                pixel = project_world_to_camera(
                    point=(detection.x, detection.y, 0.45),
                    camera_transform=frame.camera_transform,
                    width=image.shape[1],
                    height=image.shape[0],
                    fov_deg=CAMERA_FOV_DEG,
                )
                if pixel is None:
                    continue
                color = bgr_for_cone_color(detection.color)
                cv2.circle(image, pixel, 7, color, 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    detection.color,
                    (pixel[0] + 8, pixel[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Step 5: add a compact title so the window is self-identifying.
        cv2.putText(
            image,
            "camera: RGB + detected cones",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    def _lidar_view(
        self,
        frame: SensorFrame,
        clusters: Sequence[LidarCluster],
        detections: Sequence[ConeLandmark],
        landmarks: Sequence[ConeLandmark],
    ) -> np.ndarray:
        # Step 1: create a black top-down canvas centered on the LiDAR.
        canvas = np.zeros(
            (LIDAR_VIEW_SIZE_PX, LIDAR_VIEW_SIZE_PX, 3),
            dtype=np.uint8,
        )

        # Step 2: draw range grid lines so point distances are readable.
        self._draw_lidar_grid(canvas)

        raw_count = 0
        visible_count = 0
        range_text = "no lidar packet"

        # Step 3: draw raw LiDAR points in gray in sensor coordinates.
        if frame.lidar_points is not None:
            points = frame.lidar_points[:, :3]
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            raw_count = len(points)
            radius = np.linalg.norm(points[:, :2], axis=1) <= LIDAR_VIEW_RADIUS_M
            visible = points[radius]
            visible_count = len(visible)

            if raw_count:
                mins = points.min(axis=0)
                maxs = points.max(axis=0)
                range_text = (
                    f"x[{mins[0]:+.1f},{maxs[0]:+.1f}] "
                    f"y[{mins[1]:+.1f},{maxs[1]:+.1f}] "
                    f"z[{mins[2]:+.1f},{maxs[2]:+.1f}]"
                )

            stride = max(len(visible) // 9000, 1)
            for point in visible[::stride]:
                pixel = lidar_point_to_pixel(float(point[0]), float(point[1]))
                if pixel is not None:
                    cv2.circle(canvas, pixel, 1, (160, 160, 160), -1, cv2.LINE_AA)

        # Step 4: draw raw LiDAR clusters in magenta before color classification.
        if frame.lidar_transform is not None and clusters:
            local_clusters = transform_world_points_to_local(
                np.asarray([(item.x, item.y, item.z) for item in clusters], dtype=float),
                frame.lidar_transform,
            )
            for local in local_clusters:
                pixel = lidar_point_to_pixel(float(local[0]), float(local[1]))
                if pixel is not None:
                    cv2.circle(canvas, pixel, 7, (255, 0, 255), 2, cv2.LINE_AA)

        # Step 5: draw the online map as small dim points if LiDAR transform is available.
        if frame.lidar_transform is not None and landmarks:
            local_landmarks = transform_world_points_to_local(
                np.asarray([(item.x, item.y, 0.35) for item in landmarks], dtype=float),
                frame.lidar_transform,
            )
            for landmark, local in zip(landmarks, local_landmarks):
                pixel = lidar_point_to_pixel(float(local[0]), float(local[1]))
                if pixel is not None:
                    color = dim_bgr(bgr_for_cone_color(landmark.color))
                    cv2.circle(canvas, pixel, 4, color, -1, cv2.LINE_AA)

        # Step 6: draw current-frame detections as larger bright circles.
        if frame.lidar_transform is not None and detections:
            local_detections = transform_world_points_to_local(
                np.asarray([(item.x, item.y, 0.45) for item in detections], dtype=float),
                frame.lidar_transform,
            )
            for detection, local in zip(detections, local_detections):
                pixel = lidar_point_to_pixel(float(local[0]), float(local[1]))
                if pixel is not None:
                    cv2.circle(
                        canvas,
                        pixel,
                        8,
                        bgr_for_cone_color(detection.color),
                        2,
                        cv2.LINE_AA,
                    )

        # Step 7: draw the ego vehicle marker and window title.
        ego_pixel = lidar_point_to_pixel(0.0, 0.0)
        if ego_pixel is not None:
            cv2.circle(canvas, ego_pixel, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(
                canvas,
                ego_pixel,
                (ego_pixel[0], max(ego_pixel[1] - 24, 0)),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            f"lidar: raw={raw_count} visible={visible_count} "
            f"clusters={len(clusters)} det={len(detections)} map={len(landmarks)}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            range_text,
            (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _draw_lidar_grid(self, canvas: np.ndarray) -> None:
        # Step 1: draw 360-degree distance rings around the LiDAR.
        center = lidar_point_to_pixel(0.0, 0.0)
        if center is None:
            return
        margin_px = 36
        scale_px_per_m = (LIDAR_VIEW_SIZE_PX - 2 * margin_px) / (
            2.0 * LIDAR_VIEW_RADIUS_M
        )
        for distance_m in range(5, int(LIDAR_VIEW_RADIUS_M) + 1, 5):
            radius_px = int(round(distance_m * scale_px_per_m))
            cv2.circle(canvas, center, radius_px, (34, 34, 34), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{distance_m}m",
                (center[0] + radius_px + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        # Step 2: draw LiDAR local x/y axes.
        forward = lidar_point_to_pixel(LIDAR_VIEW_RADIUS_M, 0.0)
        backward = lidar_point_to_pixel(-LIDAR_VIEW_RADIUS_M, 0.0)
        left = lidar_point_to_pixel(0.0, -LIDAR_VIEW_RADIUS_M)
        right = lidar_point_to_pixel(0.0, LIDAR_VIEW_RADIUS_M)
        if forward is not None and backward is not None:
            cv2.line(canvas, backward, forward, (45, 45, 45), 1, cv2.LINE_AA)
        if left is not None and right is not None:
            cv2.line(canvas, left, right, (45, 45, 45), 1, cv2.LINE_AA)


def transform_points(points: np.ndarray, transform: carla.Transform) -> np.ndarray:
    # Step 1: convert an Nx3 array into homogeneous coordinates.
    homogeneous = np.ones((len(points), 4), dtype=float)
    homogeneous[:, :3] = points

    # Step 2: apply CARLA's local-to-world transform matrix.
    matrix = np.asarray(transform.get_matrix(), dtype=float)
    return (homogeneous @ matrix.T)[:, :3]


def transform_world_points_to_local(
    points: np.ndarray,
    transform: carla.Transform,
) -> np.ndarray:
    # Step 1: convert an Nx3 world array into homogeneous coordinates.
    homogeneous = np.ones((len(points), 4), dtype=float)
    homogeneous[:, :3] = points

    # Step 2: apply CARLA's world-to-local inverse transform matrix.
    inverse_matrix = np.asarray(transform.get_inverse_matrix(), dtype=float)
    return (homogeneous @ inverse_matrix.T)[:, :3]


def lidar_point_to_pixel(local_x_m: float, local_y_m: float) -> tuple[int, int] | None:
    # Step 1: keep only points within the circular top-down LiDAR view.
    if math.hypot(local_x_m, local_y_m) > LIDAR_VIEW_RADIUS_M:
        return None

    # Step 2: map CARLA local x-forward/y-right meters to image pixels.
    margin_px = 36
    usable_w = LIDAR_VIEW_SIZE_PX - 2 * margin_px
    usable_h = LIDAR_VIEW_SIZE_PX - 2 * margin_px
    x_px = margin_px + int(
        round((local_y_m + LIDAR_VIEW_RADIUS_M) / (2.0 * LIDAR_VIEW_RADIUS_M) * usable_w)
    )
    y_px = LIDAR_VIEW_SIZE_PX - margin_px - int(
        round((local_x_m + LIDAR_VIEW_RADIUS_M) / (2.0 * LIDAR_VIEW_RADIUS_M) * usable_h)
    )
    return (x_px, y_px)


def bgr_for_cone_color(color: str) -> tuple[int, int, int]:
    # Step 1: OpenCV drawing functions use BGR order.
    if color == "blue":
        return (255, 80, 30)
    if color == "yellow":
        return (30, 220, 255)
    if color == "orange":
        return (20, 130, 255)
    return (180, 180, 180)


def dim_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    # Step 1: dim map landmarks so current-frame detections stand out.
    return tuple(int(channel * 0.45) for channel in color)


def classify_fsae_cone_patch(rgb_patch: np.ndarray) -> str:
    # Step 1: keep this wrapper for simple call sites.
    color, count, required = score_fsae_cone_patch(rgb_patch)
    if count >= required:
        return color
    return "unknown"


def score_fsae_cone_patch(rgb_patch: np.ndarray) -> tuple[str, int, int]:
    # Step 1: HSV separates hue from brightness, which is more stable in CARLA.
    if rgb_patch.size == 0:
        return "unknown", 0, 1
    hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Step 2: count pixels in broad FSAE cone color bands.
    blue = (hue >= 90) & (hue <= 135) & (sat >= 45) & (val >= 35)
    yellow = (hue >= 18) & (hue <= 42) & (sat >= 40) & (val >= 45)
    orange = (hue >= 3) & (hue <= 24) & (sat >= 45) & (val >= 45)
    counts = {
        "blue": int(np.count_nonzero(blue)),
        "yellow": int(np.count_nonzero(yellow)),
        "orange": int(np.count_nonzero(orange)),
    }

    # Step 3: require enough colored pixels to avoid classifying plain asphalt.
    min_pixels = max(3, int(rgb_patch.shape[0] * rgb_patch.shape[1] * 0.004))
    color, count = max(counts.items(), key=lambda item: item[1])
    return color, count, min_pixels


def project_world_to_camera(
    point: tuple[float, float, float],
    camera_transform: carla.Transform,
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[int, int] | None:
    # Step 1: transform the world point into the camera frame.
    world_point = np.asarray([point[0], point[1], point[2], 1.0], dtype=float)
    world_to_camera = np.asarray(camera_transform.get_inverse_matrix(), dtype=float)
    camera_point = world_to_camera @ world_point

    # Step 2: CARLA camera x is depth; points behind the camera cannot be projected.
    depth = float(camera_point[0])
    if depth <= 0.05:
        return None

    # Step 3: use a pinhole camera model to project 3D to 2D pixels.
    focal = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    u = int(round(width * 0.5 + focal * float(camera_point[1]) / depth))
    v = int(round(height * 0.5 - focal * float(camera_point[2]) / depth))
    if 0 <= u < width and 0 <= v < height:
        return u, v
    return None


def euclidean_clusters_xy(
    points: np.ndarray,
    radius_m: float,
    min_points: int,
) -> list[np.ndarray]:
    # Step 1: index points into a grid so neighbor search stays cheap.
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        key = (math.floor(point[0] / radius_m), math.floor(point[1] / radius_m))
        cells[key].append(index)

    # Step 2: flood-fill nearby points into clusters.
    visited: set[int] = set()
    clusters: list[np.ndarray] = []
    radius_sq = radius_m * radius_m
    for start_index in range(len(points)):
        if start_index in visited:
            continue

        queue: deque[int] = deque([start_index])
        visited.add(start_index)
        cluster_indices: list[int] = []

        while queue:
            index = queue.popleft()
            cluster_indices.append(index)
            point = points[index]
            cell_x = math.floor(point[0] / radius_m)
            cell_y = math.floor(point[1] / radius_m)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for neighbor in cells.get((cell_x + dx, cell_y + dy), []):
                        if neighbor in visited:
                            continue
                        delta = points[neighbor, :2] - point[:2]
                        if float(delta @ delta) <= radius_sq:
                            visited.add(neighbor)
                            queue.append(neighbor)

        # Step 3: keep only clusters large enough to be stable cone observations.
        if len(cluster_indices) >= min_points:
            clusters.append(points[cluster_indices])
    return clusters


def attach_sensors(
    world: carla.World,
    vehicle: carla.Vehicle,
    buffer: LatestSensorBuffer,
) -> list[carla.Actor]:
    # Step 1: configure an RGB camera facing forward from the vehicle.
    blueprints = world.get_blueprint_library()
    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
    camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
    camera_bp.set_attribute("fov", str(CAMERA_FOV_DEG))

    # Step 2: configure a LiDAR with enough density to hit small cones.
    lidar_bp = blueprints.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", str(LIDAR_RANGE_M))
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("points_per_second", "240000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("upper_fov", "12")
    lidar_bp.set_attribute("lower_fov", "-32")
    lidar_bp.set_attribute("sensor_tick", "0.05")

    # Step 3: mount the camera and LiDAR on the ego car.
    camera = world.spawn_actor(
        camera_bp,
        carla.Transform(carla.Location(x=1.55, z=1.35), carla.Rotation(pitch=-5.0)),
        attach_to=vehicle,
    )
    lidar = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=0.20, z=1.65), carla.Rotation()),
        attach_to=vehicle,
    )

    # Step 4: subscribe callbacks so the control loop can read latest sensor data.
    camera.listen(buffer.update_camera)
    lidar.listen(buffer.update_lidar)
    return [camera, lidar]


def draw_landmarks(
    world: carla.World,
    landmarks: Sequence[ConeLandmark],
    lifetime_s: float = 0.25,
) -> None:
    # Step 1: visualize the online map without giving it to the planner as ground truth.
    for landmark in landmarks:
        color = carla.Color(80, 80, 80)
        if landmark.color == "blue":
            color = carla.Color(30, 80, 255)
        elif landmark.color == "yellow":
            color = carla.Color(255, 220, 30)
        elif landmark.color == "orange":
            color = carla.Color(255, 120, 20)
        world.debug.draw_point(
            carla.Location(x=landmark.x, y=landmark.y, z=0.45),
            size=0.12,
            color=color,
            life_time=lifetime_s,
        )


def print_lidar_status(
    frame: SensorFrame,
    detections: Sequence[ConeLandmark],
    landmarks: Sequence[ConeLandmark],
    raw_landmark_count: int,
    candidate_count: int,
    cluster_count: int,
    unknown_color_count: int,
    ground_z: float,
    projected_count: int,
    colored_patch_count: int,
) -> None:
    # Step 1: make missing sensor packets obvious in the terminal.
    if frame.lidar_points is None:
        print(
            f"[lidar] waiting for packet candidate={candidate_count} "
            f"clusters={cluster_count} projected={projected_count} "
            f"colored_patch={colored_patch_count} "
            f"unknown_color={unknown_color_count} det={len(detections)} "
            f"raw_map={raw_landmark_count} stable_map={len(landmarks)}"
        )
        return

    # Step 2: report raw and visible point counts using the same view filter as the window.
    points = frame.lidar_points[:, :3]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    radius = np.linalg.norm(points[:, :2], axis=1) <= LIDAR_VIEW_RADIUS_M
    visible = points[radius]
    if len(points) == 0:
        print(
            f"[lidar] raw=0 visible=0 ground_z={ground_z:+.2f} "
            f"candidate={candidate_count} clusters={cluster_count} "
            f"projected={projected_count} colored_patch={colored_patch_count} "
            f"unknown_color={unknown_color_count} det={len(detections)} "
            f"raw_map={raw_landmark_count} stable_map={len(landmarks)}"
        )
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(
        "[lidar] "
        f"raw={len(points)} visible={len(visible)} "
        f"ground_z={ground_z:+.2f} candidate={candidate_count} "
        f"clusters={cluster_count} projected={projected_count} "
        f"colored_patch={colored_patch_count} "
        f"unknown_color={unknown_color_count} "
        f"det={len(detections)} raw_map={raw_landmark_count} "
        f"stable_map={len(landmarks)} "
        f"x=[{mins[0]:+.1f},{maxs[0]:+.1f}] "
        f"y=[{mins[1]:+.1f},{maxs[1]:+.1f}] "
        f"z=[{mins[2]:+.1f},{maxs[2]:+.1f}]"
    )


def speed_limited_by_visible_path(
    target_speed_mps: float,
    reference_index: int,
    trajectory_length: int,
    remaining_path_m: float,
) -> float:
    # Step 1: if the visible local path is almost exhausted, slow down before the unknown.
    if trajectory_length - reference_index <= 4:
        return min(target_speed_mps, MIN_SPEED_MPS)

    # Step 2: otherwise cap speed by what can be stopped within the current sensed path.
    braking_speed = math.sqrt(max(2.0 * MAX_BRAKE_MPS2 * remaining_path_m, 0.0))
    if not math.isfinite(target_speed_mps):
        return max(MIN_SPEED_MPS, braking_speed)
    return min(target_speed_mps, max(MIN_SPEED_MPS, braking_speed))


def main() -> None:
    # Step 1: connect to CARLA and create the same flat FSAE-style test world.
    client = setup_client()
    world = setup_world(client)
    print("[sensor] connected and generated blank CARLA world")

    # Step 2: generate and spawn cones; after this line the planner never reads track.cones.
    generator = TrackGenerator(seed=42)
    track = generator.generate_autocross(
        name="sensor_autocross",
        length=260.0,
        width=6.0,
        curvature=0.35,
        cone_density=0.65,
    )
    mesh = generator.build_mesh_spec(track)
    carla_spec = generator.build_carla_import_spec(track, mesh)
    cone_actors = spawn_cones_in_carla(world, carla_spec)
    start_x, start_y, start_heading = carla_spec.start_pose
    car = spawn_car(world, start_x, start_y, start_heading)
    print(f"[sensor] spawned {len(cone_actors)} cones as world objects only")

    # Step 3: attach sensors and create online perception, mapping, planning, and control blocks.
    sensor_buffer = LatestSensorBuffer()
    sensor_actors = attach_sensors(world, car, sensor_buffer)
    detector = CameraLidarConeDetector()
    cone_map = OnlineConeMap()
    viewer = SensorDebugViewer()
    planner = DelaunayMinimumCurvaturePlanner(
        expected_track_width_m=6.0,
        waypoint_spacing_m=1.5,
    )
    lateral_mpc = LateralMPCController()
    longitudinal_pid = LongitudinalPID()

    # Step 4: keep controller memory between simulator ticks.
    last_steer_rad = 0.0
    last_time = time.monotonic()
    last_print_time = last_time
    last_draw_time = last_time
    print("[sensor] running LiDAR/camera perception -> online map -> Delaunay -> MPC")

    try:
        while True:
            # Step 5: compute the controller timestep.
            now = time.monotonic()
            dt_s = clamp(now - last_time, 1e-3, 0.2)
            last_time = now

            # Step 6: read localization from simulator ground truth, but not cone locations.
            state = read_vehicle_state(car, last_steer_rad)

            # Step 7: detect cones from the latest sensor packets and merge them into the map.
            frame = sensor_buffer.snapshot()
            detections = detector.detect(frame)
            landmarks = cone_map.update(detections)
            raw_landmark_count = len(cone_map.raw_landmarks)
            viewer.show(frame, detector.last_clusters, detections, landmarks)

            # Step 8: wait until enough blue/yellow landmarks exist for a stable Delaunay path.
            boundary_landmarks = [
                landmark
                for landmark in landmarks
                if landmark.color in {"blue", "yellow"}
            ]
            if len(boundary_landmarks) < MIN_LANDMARKS_FOR_PLANNING:
                car.apply_control(
                    carla.VehicleControl(throttle=0.12, steer=0.0, brake=0.0)
                )
                if now - last_print_time >= 1.0:
                    print_lidar_status(
                        frame,
                        detections,
                        landmarks,
                        raw_landmark_count,
                        detector.last_candidate_count,
                        detector.last_cluster_count,
                        detector.last_unknown_color_count,
                        detector.last_ground_z,
                        detector.last_projected_cluster_count,
                        detector.last_colored_patch_count,
                    )
                    last_print_time = now
                world.wait_for_tick()
                continue

            # Step 9: plan a local centerline from the online map, starting at the car pose.
            planned_path = planner.plan(
                boundary_landmarks,
                start=(state.x, state.y),
                start_yaw_rad=state.yaw_rad,
                closed=False,
            )
            if len(planned_path.trajectory) < MIN_PATH_POINTS_FOR_CONTROL:
                car.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.35)
                )
                if now - last_print_time >= 1.0:
                    print_lidar_status(
                        frame,
                        detections,
                        landmarks,
                        raw_landmark_count,
                        detector.last_candidate_count,
                        detector.last_cluster_count,
                        detector.last_unknown_color_count,
                        detector.last_ground_z,
                        detector.last_projected_cluster_count,
                        detector.last_colored_patch_count,
                    )
                    last_print_time = now
                world.wait_for_tick()
                continue

            # Step 10: convert the planned centerline into x_ref/y_ref/heading/curvature/v_ref.
            reference_trajectory = build_reference_trajectory(
                centerline=planned_path.trajectory,
                closed=False,
                spacing_m=1.0,
                min_speed_mps=MIN_SPEED_MPS,
                max_speed_mps=MAX_SPEED_MPS,
                max_lateral_accel_mps2=MAX_LATERAL_ACCEL_MPS2,
                max_accel_mps2=MAX_ACCEL_MPS2,
                max_brake_mps2=MAX_BRAKE_MPS2,
            )

            # Step 11: choose a lookahead speed from the reference trajectory.
            nearest_index = nearest_reference_index(
                reference_trajectory,
                state.x,
                state.y,
            )
            speed_lookahead_m = clamp(
                state.speed_mps * 1.4 + 6.0,
                MIN_SPEED_LOOKAHEAD_M,
                MAX_SPEED_LOOKAHEAD_M,
            )
            target_speed_mps = target_speed_ahead(
                trajectory=reference_trajectory,
                nearest_index=nearest_index,
                lookahead_m=speed_lookahead_m,
                closed=False,
            )

            # Step 12: limit speed by the currently visible path length, not by a fixed top speed.
            remaining_path_m = max(
                reference_trajectory[-1].s_m - reference_trajectory[nearest_index].s_m,
                0.0,
            )
            target_speed_mps = speed_limited_by_visible_path(
                target_speed_mps,
                nearest_index,
                len(reference_trajectory),
                remaining_path_m,
            )

            # Step 13: use all mapped cone positions as clearance obstacles for lateral MPC.
            cone_points: list[Point2] = [(landmark.x, landmark.y) for landmark in landmarks]
            lateral_command = lateral_mpc.control(
                state=state,
                trajectory=reference_trajectory,
                cone_points=cone_points,
                closed=False,
            )

            # Step 14: use PID to convert the target speed into throttle/brake.
            throttle, brake = longitudinal_pid.control(
                target_speed_mps=target_speed_mps,
                current_speed_mps=state.speed_mps,
                dt_s=dt_s,
            )

            # Step 15: send the combined longitudinal and lateral command to CARLA.
            car.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=lateral_command.steer,
                    brake=brake,
                    hand_brake=False,
                    reverse=False,
                )
            )
            last_steer_rad = lateral_command.steer_rad

            # Step 16: draw and print low-rate diagnostics so the simulator stays responsive.
            if now - last_draw_time >= 0.25:
                draw_landmarks(world, landmarks)
                draw_reference_trajectory(world, reference_trajectory, lifetime_s=0.3)
                last_draw_time = now

            if now - last_print_time >= 1.0:
                ref = reference_trajectory[nearest_index]
                cross_track_error = signed_lateral_error(state.x, state.y, ref)
                print(
                    "[sensor] "
                    f"candidate={detector.last_candidate_count} "
                    f"clusters={detector.last_cluster_count} "
                    f"projected={detector.last_projected_cluster_count} "
                    f"colored_patch={detector.last_colored_patch_count} "
                    f"unknown_color={detector.last_unknown_color_count} "
                    f"det={len(detections)} raw_map={raw_landmark_count} "
                    f"stable_map={len(landmarks)} "
                    f"path={len(reference_trajectory)} "
                    f"speed={state.speed_mps:.2f}/{format_speed(target_speed_mps)} m/s "
                    f"steer={lateral_command.steer:+.2f} "
                    f"thr={throttle:.2f} brake={brake:.2f} "
                    f"cte={cross_track_error:+.2f} m"
                )
                last_print_time = now

            # Step 17: advance the simulator before the next perception-control cycle.
            world.wait_for_tick()

    except KeyboardInterrupt:
        # Step 18: stop the car cleanly when the run is interrupted.
        car.apply_control(
            carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        )
        print("[sensor] stopped")
    finally:
        # Step 19: stop sensors before destroying them to avoid dangling callbacks.
        viewer.close()
        for actor in sensor_actors:
            actor.stop()
            actor.destroy()


if __name__ == "__main__":
    main()
