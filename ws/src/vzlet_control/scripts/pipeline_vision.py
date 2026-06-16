import importlib
import math
from pathlib import Path
from typing import Optional, Tuple

from datetime import datetime
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from tf2_geometry_msgs import do_transform_point

from pipeline_types import (
    DEFAULT_YOLO_MODEL_FILE,
)


class VisionProcessor:
    def __init__(self, node):
        self.node = node
        self._yolo_model = None
        self.yolo_target_class: Optional[str] = None
        self._last_yolo_debug_boxes = []
        self.latest_bgr = None
        self.latest_image_seq = 0
        self.latest_image_header = None

    def now_s(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def load_yolo(self) -> Optional[type]:
        try:
            module = importlib.import_module("ultralytics")
        except Exception:
            return None
        return getattr(module, "YOLO", None)

    def _box_to_xyxy(self, box):
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
        return float(x1), float(y1), float(x2), float(y2)

    def _box_center_radius_area(self, box):
        x1, y1, x2, y2 = self._box_to_xyxy(box)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        center_u = 0.5 * (x1 + x2)
        center_v = 0.5 * (y1 + y2)
        radius_px = 0.5 * max(w, h)
        area = w * h
        return center_u, center_v, radius_px, area

    def _box_conf(self, box) -> float:
        return float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0

    def _box_cls_id(self, box) -> int:
        return int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
    

    # TODO: refactor it
    def pixel_to_base_parallel_camera(self, u: float, v: float):
        base_frame = self.node.get_parameter("base_frame").value
        camera_frame = self.node.get_parameter("camera_frame").value

        fx = self.fx()
        fy = self.fy()
        cx = self.cx()
        cy = self.cy()

        depth_m = self.node.motion.get_camera_height_depth_estimate()

        self.node.get_logger().info(
            f"YOLO pixel-to-base input: "
            f"u={u:.1f}, v={v:.1f}, "
            f"fx={fx:.1f}, fy={fy:.1f}, "
            f"cx={cx:.1f}, cy={cy:.1f}, "
            f"depth_m={depth_m:.4f}"
        )

        if depth_m is None or depth_m <= 0.0:
            self.node.get_logger().warn("Invalid camera-to-plane depth")
            return None

        try:
            # Pixel -> metric point in camera optical frame
            target_cam = PointStamped()
            target_cam.header.frame_id = camera_frame
            target_cam.header.stamp = self.node.get_clock().now().to_msg()

            target_cam.point.x = (u - cx) * depth_m / fx
            target_cam.point.y = (v - cy) * depth_m / fy
            target_cam.point.z = depth_m

            # Camera frame -> base_link
            tf_base_cam = self.node.tf_buffer.lookup_transform(
                base_frame,
                camera_frame,
                rclpy.time.Time(),
            )
            self.node.get_logger().info(f"target_cam: {target_cam}")
            target_base = do_transform_point(target_cam, tf_base_cam)

            # Object is assumed to lie on the known target plane
            target_base.point.z = float(
                self.node.get_parameter("target_plane_z_base").value
            )

            return target_base

        except Exception as exc:
            self.node.get_logger().warn(
                f"Could not convert YOLO pixel to base point: {exc}"
            )
            return None



    def _get_next_bgr_frame(self, last_seq: int, timeout_s: float):
        start_s = self.now_s()

        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)

            if self.latest_bgr is not None and self.latest_image_seq != last_seq:
                return (
                    self.latest_bgr.copy(),
                    self.latest_image_seq,
                    self.latest_image_header,
                )

            if (self.now_s() - start_s) > timeout_s:
                return None, last_seq, None

        return None, last_seq, None


    def _detect_yolo_candidates(
        self,
        bgr: np.ndarray,
        target_class_name: str,
        min_conf: float,
    ):
        model = self.get_yolo_model()
        if model is None:
            return None

        try:
            results = model.predict(source=bgr, verbose=False, conf=0.15)
        except Exception as exc:
            self.node.get_logger().warn(f"YOLO inference failed: {exc}")
            return None

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return []

        target_class_id = self._get_target_class_id(result, target_class_name)
        if target_class_id is None:
            return None

        matched_boxes = self._matched_boxes(boxes, target_class_id)

        candidates = [
            self._make_debug_candidate(box, target_class_name)
            for box in matched_boxes
        ]

        return [
            candidate for candidate in candidates
            if candidate["conf"] > min_conf
        ]


    # TODO: refactor it
    def _match_candidates_to_reference(
        self,
        reference_candidates,
        current_candidates,
        max_center_dist_px: float,
    ):
        if len(reference_candidates) != len(current_candidates):
            return None

        unused = list(current_candidates)
        matched = []

        for ref in reference_candidates:
            best_index = None
            best_dist = None

            for index, candidate in enumerate(unused):
                dist = math.hypot(
                    candidate["center_u"] - ref["center_u"],
                    candidate["center_v"] - ref["center_v"],
                )

                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_index = index

            if best_index is None or best_dist > max_center_dist_px:
                return None

            matched.append(unused.pop(best_index))

        return matched

    # TODO: refactor it
    def _average_candidate_votes(self, candidate_votes):
        averaged = []

        candidate_count = len(candidate_votes[0])

        for candidate_index in range(candidate_count):
            samples = [
                frame_candidates[candidate_index]
                for frame_candidates in candidate_votes
            ]

            base = dict(samples[0])
            base["center_u"] = sum(item["center_u"] for item in samples) / len(samples)
            base["center_v"] = sum(item["center_v"] for item in samples) / len(samples)
            base["radius_px"] = sum(item["radius_px"] for item in samples) / len(samples)
            base["area"] = sum(item["area"] for item in samples) / len(samples)
            base["conf"] = sum(item["conf"] for item in samples) / len(samples)
            base["selected"] = False

            averaged.append(base)

        return averaged


    def _draw_vote_debug_image(
    self,
    bgr: np.ndarray,
    candidates,
    target_class_name: str,
    vote_index: int,
    vote_frames: int,
    ):
        image = bgr.copy()

        for candidate in candidates:
            x1, y1, x2, y2 = candidate["xyxy"]
            conf = candidate["conf"]
            class_name = candidate.get("class_name", target_class_name)

            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))

            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)

            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                image,
                label,
                (p1[0], max(20, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            center = (
                int(round(candidate["center_u"])),
                int(round(candidate["center_v"])),
            )
            cv2.drawMarker(
                image,
                center,
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
            )

        status = (
            f"YOLO vote frame {vote_index}/{vote_frames}: "
            f"class={target_class_name}, candidates={len(candidates)}"
        )

        cv2.putText(
            image,
            status,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return image

    def _create_vote_debug_run_dir(self):
        root_dir = Path(str(self.node.get_parameter("yolo_vote_debug_dir").value))
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_dir = root_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_vote_debug_image(self, debug_image, run_dir: Path, vote_index: int):
        image_path = run_dir / f"{vote_index}.jpg"

        try:
            cv2.imwrite(str(image_path), debug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as exc:
            self.node.get_logger().warn(
                f"Could not save YOLO vote debug image {image_path}: {exc}"
            )

    # TODO: refactor it
    def get_stable_yolo_candidates(
    self,
    target_class_name: str,
    ):
        vote_frames = int(self.node.get_parameter("yolo_vote_frames").value)
        max_center_dist_px = float(
            self.node.get_parameter("yolo_vote_max_center_dist_px").value
        )
        min_conf = float(self.node.get_parameter("yolo_vote_min_conf").value)
        frame_timeout_s = float(
            self.node.get_parameter("yolo_vote_frame_timeout_s").value
        )

        if vote_frames < 1:
            vote_frames = 1

        candidate_votes = []
        last_seq = self.latest_image_seq
        run_dir = self._create_vote_debug_run_dir()

        self.node.get_logger().info(
            f"Starting YOLO vote: target_class={target_class_name}, "
            f"frames={vote_frames}, min_conf={min_conf:.2f}, "
            f"max_center_dist_px={max_center_dist_px:.1f}, "
            f"debug_dir={run_dir}"
        )

        for vote_index in range(vote_frames):
            frame_number = vote_index + 1

            bgr, last_seq, header = self._get_next_bgr_frame(
                last_seq=last_seq,
                timeout_s=frame_timeout_s,
            )

            if bgr is None:
                self.node.get_logger().error(
                    f"YOLO vote failed: timeout waiting for frame "
                    f"{frame_number}/{vote_frames}"
                )
                return None

            candidates = self._detect_yolo_candidates(
                bgr=bgr,
                target_class_name=target_class_name,
                min_conf=min_conf,
            )

            if candidates is None:
                return None

            debug_image = self._draw_vote_debug_image(
                bgr=bgr,
                candidates=candidates,
                target_class_name=target_class_name,
                vote_index=frame_number,
                vote_frames=vote_frames,
            )

            self._save_vote_debug_image(debug_image, run_dir, frame_number)

            self.node.get_logger().info(
                f"YOLO vote frame {frame_number}/{vote_frames}: "
                f"candidates={len(candidates)}, "
                f"detections={[(round(c['center_u'], 1), round(c['center_v'], 1), round(c['conf'], 3)) for c in candidates]}"
            )

            if not candidates:
                self.node.get_logger().error(
                    f"YOLO vote failed: no '{target_class_name}' candidates in frame "
                    f"{frame_number}/{vote_frames}"
                )
                return None

            if vote_index == 0:
                candidate_votes.append(candidates)
                continue

            matched = self._match_candidates_to_reference(
                reference_candidates=candidate_votes[0],
                current_candidates=candidates,
                max_center_dist_px=max_center_dist_px,
            )

            if matched is None:
                self.node.get_logger().error(
                    f"YOLO vote failed: frame {frame_number}/{vote_frames} does not "
                    f"match reference frame. Expected count={len(candidate_votes[0])}, "
                    f"got count={len(candidates)}, max_center_dist_px={max_center_dist_px:.1f}. "
                    f"reference={[(round(c['center_u'], 1), round(c['center_v'], 1), round(c['conf'], 3)) for c in candidate_votes[0]]}, "
                    f"current={[(round(c['center_u'], 1), round(c['center_v'], 1), round(c['conf'], 3)) for c in candidates]}"
                )
                return None

            candidate_votes.append(matched)

        averaged_candidates = self._average_candidate_votes(candidate_votes)

        self.node.get_logger().info(
            f"YOLO vote accepted: target_class={target_class_name}, "
            f"frames={vote_frames}, candidates={len(averaged_candidates)}, "
            f"debug_dir={run_dir}"
        )

        return averaged_candidates

    # TODO: refactor it
    def select_yolo_grid_pose(
        self,
        target_class_name: str,
    ):
        candidates = self.get_stable_yolo_candidates(target_class_name)

        if not candidates:
            return None

        tool_projection = self.node.motion.project_tool0_to_image()

        selected_candidate = self._select_nearest_to_tool(
            candidates,
            tool_projection,
        )
        selected_candidate["selected"] = True
        self._last_yolo_debug_boxes = candidates

        target_base = self.pixel_to_base_parallel_camera(
            selected_candidate["center_u"],
            selected_candidate["center_v"],
        )

        if target_base is None:
            self.node.get_logger().error(
                "Could not convert selected YOLO pixel to base_link XY"
            )
            return None

        pose = self.node.motion.find_closest_zone_pose_to_base_xy(
            target_base.point.x,
            target_base.point.y,
        )

        if pose is None:
            self.node.get_logger().error(
                "Could not match selected YOLO candidate to any CSV grid pose"
            )
            return None

        self.node.get_logger().info(
            f"Selected YOLO {target_class_name}: "
            f"u={selected_candidate['center_u']:.1f}, "
            f"v={selected_candidate['center_v']:.1f}, "
            f"base_x={target_base.point.x:.4f}, "
            f"base_y={target_base.point.y:.4f}, "
            f"base_z={target_base.point.z:.4f}, "
            f"matched grid pose={pose['name']}"
        )

        return pose["name"]

    def image_callback(self, msg: Image):
        try:
            bgr = self.node.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.node.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        self.latest_bgr = bgr
        self.latest_image_seq += 1
        self.latest_image_header = msg.header
       
    def _get_target_class_id(self, result, target_class_name: str):
        names = getattr(result, "names", {})

        for cls_id, cls_name in names.items():
            if cls_name == target_class_name:
                return int(cls_id)

        self.node.get_logger().warn(
            f"YOLO target class '{target_class_name}' not found in model classes: {names}"
        )
        return None

    def _matched_boxes(self, boxes, target_class_id: int):
        return [
            box for box in boxes
            if self._box_cls_id(box) == target_class_id
        ]

    def _make_debug_candidate(self, box, class_name: str):
        conf = self._box_conf(box)
        center_u, center_v, radius_px, area = self._box_center_radius_area(box)
        x1, y1, x2, y2 = self._box_to_xyxy(box)

        return {
            "box": box,
            "xyxy": (x1, y1, x2, y2),
            "conf": conf,
            "center_u": center_u,
            "center_v": center_v,
            "radius_px": radius_px,
            "area": area,
            "selected": False,
            "class_name": class_name,
        }

    def _select_highest_confidence(self, candidates):
        return max(candidates, key=lambda item: item["conf"])


    def _select_nearest_to_tool(self, candidates, tool_projection):
        if tool_projection is None:
            return self._select_highest_confidence(candidates)

        tool0_u, tool0_v, _tool0_z_cam = tool_projection

        return min(
            candidates,
            key=lambda item: math.hypot(
                item["center_u"] - tool0_u,
                item["center_v"] - tool0_v,
            ),
        )

    def get_yolo_model(self):
        if self._yolo_model is not None:
            return self._yolo_model

        yolo_class = self.load_yolo()
        if yolo_class is None:
            self.node.get_logger().error("Ultralytics is not available, cannot use neural circle detection")
            return None

        model_path = Path(str(self.node.get_parameter("yolo_model_path").value or DEFAULT_YOLO_MODEL_FILE))
        if not model_path.exists():
            self.node.get_logger().error(f"YOLO model file not found: {model_path}")
            return None

        try:
            self._yolo_model = yolo_class(str(model_path))
            self.node.get_logger().info(f"Loaded YOLO circle detector from {model_path}")
            return self._yolo_model
        except Exception as exc:
            self.node.get_logger().error(f"Could not load YOLO model {model_path}: {exc}")
            return None

    # TODO: refactor it
    def fx(self):
        return float(self.node.get_parameter("fx").value)

    def fy(self):
        return float(self.node.get_parameter("fy").value)

    def cx(self):
        return float(self.node.get_parameter("cx").value)

    def cy(self):
        return float(self.node.get_parameter("cy").value)