import importlib
import math
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from tf2_geometry_msgs import do_transform_point

from pipeline_types import (
    CIRCLE_DETECTION_CONTOUR,
    CIRCLE_DETECTION_MASK,
    CIRCLE_DETECTION_YOLO,
    DEFAULT_YOLO_MODEL_FILE,
)


class VisionProcessor:
    def __init__(self, node):
        self.node = node
        self._yolo_model = None
        self.yolo_target_class: Optional[str] = None
        # Visual servo tracking states
        self.visual_servo_active = False
        self.visual_servo_centered = False
        self.centered_since_s: Optional[float] = None
        self.last_target_time_s: Optional[float] = None
        self.last_pixel_error: Optional[Tuple[float, float]] = None
        self._servo_publish_active = False
        self._last_yolo_debug_boxes = []
        self._last_tool_projection: Optional[Tuple[float, float, float]] = None

    def now_s(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def load_yolo(self) -> Optional[type]:
        try:
            module = importlib.import_module("ultralytics")
        except Exception:
            return None
        return getattr(module, "YOLO", None)

    def run_visual_servo_stage(
        self,
        method: Optional[str] = None,
        yolo_target_class: Optional[str] = None,
    ) -> bool:
        # Dynamically switch tracking configuration right before starting the servo loop
        if method is not None:
            self.node.get_logger().info(f"Setting pipeline tracking parameter to: {method}")
            self.node.set_parameters([Parameter("circle_detection_method", value=method)])

        self.yolo_target_class = yolo_target_class
        if yolo_target_class is not None:
            self.node.get_logger().info(f"Setting YOLO target class to: {yolo_target_class}")

        if not self.node.motion.switch_to_servo_controller():
            return False

        if not self.node.motion.set_servo_command_type():
            self.node.motion.switch_to_trajectory_controller()
            return False
            
        self.visual_servo_active = True
        self.visual_servo_centered = False
        self.centered_since_s = None
        self.last_target_time_s = None
        self.last_pixel_error = None

        # --- High-frequency background publisher thread --- #TODO: really works? no bad discrete movements?
        self.node.motion.target_twist = None
        self.node.motion.last_vision_update_s = self.now_s()
        self._servo_publish_active = True

        def continuous_publish_loop():
            # 50 Hz continuous publishing loop to keep MoveIt Servo smooth
            rate_sleep = 1.0 / 50.0
            while self._servo_publish_active and rclpy.ok():
                now = self.now_s()
                if self.node.motion.target_twist is not None:
                    # If YOLO hasn't updated in 1.0 second, stop the robot for safety
                    if (now - self.node.motion.last_vision_update_s) > 1.0:
                        self.node.motion.publish_zero_twist(reason="stale vision")
                    else:
                        # Re-stamp and publish the continuous velocity
                        self.node.motion.target_twist.header.stamp = self.node.get_clock().now().to_msg()
                        self.node.twist_pub.publish(self.node.motion.target_twist)
                time.sleep(rate_sleep)

        publisher_thread = threading.Thread(target=continuous_publish_loop, daemon=True)
        publisher_thread.start()
        # -------------------------------------------------------

        timeout_s = float(self.node.get_parameter("servo_timeout").value)
        settle_time_s = float(self.node.get_parameter("servo_settle_time").value)
        start_s = self.now_s()
        servo_ok = False

        self.node.get_logger().info(
            f"Visual servo active: timeout={timeout_s:.1f}s, settle_time={settle_time_s:.2f}s"
        )

        try:
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)
                now_s = self.now_s()

                if self.visual_servo_centered:
                    self.node.motion.publish_zero_twist(reason="centered")
                    self.node.get_logger().info("Visual servo complete: target centered")
                    servo_ok = True
                    return True

                if (now_s - start_s) > timeout_s:
                    self.node.motion.publish_zero_twist(reason="servo timeout")
                    if self.last_pixel_error is None:
                        self.node.get_logger().error(
                            "Visual servo timed out before any valid target was tracked"
                        )
                    else:
                        du, dv = self.last_pixel_error
                        self.node.get_logger().error(
                            f"Visual servo timed out: last pixel error du={du:+.1f}, dv={dv:+.1f}"
                        )
                    return False
        finally:
            self.visual_servo_active = False
            self._servo_publish_active = False
            publisher_thread.join(timeout=1.0)
            self.node.motion.publish_zero_twist(reason="visual servo inactive")

            switched_back = self.node.motion.switch_to_trajectory_controller()
            if servo_ok and not switched_back:
                self.node.get_logger().error(
                    "Visual servo converged, but switching back to joint_trajectory_controller failed"
                )

        return False

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
    

    def pixel_to_base_parallel_camera(self, u: float, v: float):
        base_frame = self.node.get_parameter("base_frame").value
        camera_frame = self.node.get_parameter("camera_frame").value

        fx = self.fx()
        fy = self.fy()
        cx = self.cx()
        cy = self.cy()

        depth_m = self.node.motion.get_camera_height_depth_estimate()
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


    def image_callback(self, msg: Image):
        if not self.visual_servo_active:
            return

        try:
            bgr = self.node.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.node.get_logger().warn(f"cv_bridge conversion failed: {exc}")
            return

        debug_image = bgr.copy()

        self._last_tool_projection = self.node.motion.project_tool0_to_image()

        detection = self.detect_target_circle(bgr)

        if detection is None:
            self.draw_no_target_overlay(debug_image)
            self.publish_debug_image(debug_image, msg.header)
            self.centered_since_s = None
            if bool(self.node.get_parameter("stop_when_lost").value):
                self.node.motion.publish_zero_twist(reason="target lost")
            return
        
        target_u, target_v, radius_px = detection
        target_base = self.pixel_to_base_parallel_camera( target_u, target_v, )

        tool_pose = self.node.motion.get_current_tool0_pose()

        depth_m = self.node.motion.get_camera_height_depth_estimate() or 0.0

        if target_base is None or tool_pose is None:
            self.draw_servo_overlay(
                debug_image,
                target_u,
                target_v,
                radius_px,
                target_base,
                tool_pose,
                0.0,
                0.0,
                None,
                depth_m,
                None,
                "metric target/tool unavailable",
            )
            self.publish_debug_image(debug_image, msg.header)
            self.centered_since_s = None
            self.node.motion.publish_zero_twist(reason="metric target/tool unavailable")
            return

        raw_error_x_base = target_base.point.x - tool_pose.pose.position.x
        raw_error_y_base = target_base.point.y - tool_pose.pose.position.y

        metric_deadband = 0.001  # 1 mm, tune
        error_x_base = raw_error_x_base
        error_y_base = raw_error_y_base

        if abs(error_x_base) < metric_deadband:
            error_x_base = 0.0
        if abs(error_y_base) < metric_deadband:
            error_y_base = 0.0

        error_tool = self.node.motion.transform_error_from_base_to_tool_frame(
            error_x_base,
            error_y_base,
            0.0,
            msg.header.stamp,
        )

        if error_tool is None:
            self.draw_servo_overlay(
                debug_image,
                target_u,
                target_v,
                radius_px,
                target_base,
                tool_pose,
                raw_error_x_base,
                raw_error_y_base,
                None,
                depth_m,
                None,
                "base to tool TF unavailable",
            )
            self.publish_debug_image(debug_image, msg.header)
            self.centered_since_s = None
            self.node.motion.publish_zero_twist(reason="base to tool TF unavailable")
            return

        ex_tool, ey_tool, ez_tool_observed = error_tool

        vx, vy, vz = self.node.motion.publish_servo_twist(
            ex_tool,
            ey_tool,
            0.0,
            ez_tool_observed,
            raw_error_x_base,
            raw_error_y_base,
            depth_m,
        )

        self.update_servo_convergence(
            raw_error_x_base,
            raw_error_y_base,
        )

        self.draw_servo_overlay(
            debug_image,
            target_u,
            target_v,
            radius_px,
            target_base,
            tool_pose,
            raw_error_x_base,
            raw_error_y_base,
            error_tool,
            depth_m,
            (vx, vy, vz),
            "tracking metric XY only",
        )

        self.publish_debug_image(debug_image, msg.header)
       


    def update_servo_convergence(self, raw_error_x_m: float, raw_error_y_m: float):
        now_s = self.now_s()
        deadband_m = 0.001  # 1 mm, tune
        settle_time_s = float(self.node.get_parameter("servo_settle_time").value)

        self.last_target_time_s = now_s
        self.last_pixel_error = (raw_error_x_m, raw_error_y_m)

        inside = abs(raw_error_x_m) <= deadband_m and abs(raw_error_y_m) <= deadband_m

        if inside:
            if self.centered_since_s is None:
                self.centered_since_s = now_s
            elif (now_s - self.centered_since_s) >= settle_time_s:
                self.visual_servo_centered = True
        else:
            self.centered_since_s = None
            self.visual_servo_centered = False

    def detect_target_circle(self, bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        method = str(self.node.get_parameter("circle_detection_method").value).strip().lower()
        
        if method == CIRCLE_DETECTION_YOLO:
            detection = self.detect_yolo(bgr)
            if detection is not None:
                return detection

            self.node.get_logger().warn(
                "Neural detection selected but no YOLO target was found; fallback to contour/mask"
            )
            return self.detect_circle_mask(bgr)
            # return self.detect_contour(bgr)
            
        elif method == CIRCLE_DETECTION_MASK:
            return self.detect_circle_mask(bgr)
            
        elif method == CIRCLE_DETECTION_CONTOUR:
            return self.detect_contour(bgr)
            
        else:
            self.node.get_logger().warn(
                f"Unknown circle_detection_method={method!r}; using contour detector"
            )
            return self.detect_contour(bgr)

    def detect_circle_mask(self, bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Black mask:
        # H: any hue
        # S: any saturation
        # V: low brightness
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 70])

        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Slightly larger noise gate than before, because black masks often catch shadows/noise.
            if area > max_area and area > 50.0:
                max_area = area
                best_contour = contour

        if best_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            return float(x), float(y), float(radius)

        return None
        
    # def detect_circle_mask(self, bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    #     hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
    #     # Hardcoded range for standard blue markers/circles
    #     lower_blue = np.array([100, 70, 50])
    #     upper_blue = np.array([140, 255, 255])
        
    #     mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
    #     # Morphological cleaning to strip away minor floating camera artifacts
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     best_contour = None
    #     max_area = 0.0
        
    #     # Extract the prominent blue object inside the camera viewport
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area > max_area and area > 30.0:  # Area noise gate
    #             max_area = area
    #             best_contour = contour
                
    #     if best_contour is not None:
    #         (x, y), radius = cv2.minEnclosingCircle(best_contour)
    #         return float(x), float(y), float(radius)
            
    #     return None


    def detect_contour(self, bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))
        enhanced = cv2.morphologyEx(gray_blur, cv2.MORPH_TOPHAT, tophat_kernel)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
        otsu_threshold, binary = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, clean_kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_radius = float(self.node.get_parameter("min_radius_px").value)
        max_radius = float(self.node.get_parameter("max_radius_px").value)
        min_circularity = float(self.node.get_parameter("min_circularity").value)
        min_mean_brightness = float(self.node.get_parameter("min_mean_brightness").value)

        best = None
        best_score = -1.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 150.0:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 1.0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius or radius > max_radius:
                continue
            if circularity < min_circularity:
                continue
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if w <= 1.0 or h <= 1.0:
                continue
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 1.45:
                continue
            contour_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_brightness = cv2.mean(gray, mask=contour_mask)[0]
            if mean_brightness < min_mean_brightness:
                continue
            radius_target = 28.0
            radius_error = abs(radius - radius_target)
            score = 4.0 * circularity + 0.004 * mean_brightness - 0.04 * radius_error
            if score > best_score:
                best_score = score
                best = (float(x), float(y), float(radius))

        self.node.get_logger().debug(
            f"Tophat+Otsu threshold={otsu_threshold:.1f}, contours={len(contours)}, best={best}"
        )
        return best


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


    def detect_yolo(self, bgr: np.ndarray, tool_projection: Optional[Tuple[float, float, float]] = None,) -> Optional[Tuple[float, float, float]]:
        model = self.get_yolo_model()
        if model is None:
            return None

        try:
            results = model.predict(source=bgr, verbose=False, conf=0.25)
        except Exception as exc:
            self.node.get_logger().warn(f"YOLO inference failed: {exc}")
            return None

        if not results:
            self._last_yolo_debug_boxes = []
            return None

        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            self._last_yolo_debug_boxes = []
            return None

        target_class_name = self.yolo_target_class or "sensor"
        target_class_id = self._get_target_class_id(result, target_class_name)

        if target_class_id is None:
            self._last_yolo_debug_boxes = []
            return None

        matched_boxes = self._matched_boxes(boxes, target_class_id)

        if not matched_boxes:
            self._last_yolo_debug_boxes = []
            return None

        candidates = [
            self._make_debug_candidate(box, target_class_name)
            for box in matched_boxes
        ]

        if target_class_name == "sensor":
            candidates = [
                candidate for candidate in candidates
                if candidate["conf"] > 0.5 and candidate["area"] > 50.0
            ]

            if not candidates:
                self._last_yolo_debug_boxes = []
                return None

            effective_tool_projection = (
                tool_projection
                if tool_projection is not None
                else self._last_tool_projection
            )

            selected = self._select_nearest_to_tool(
                candidates,
                effective_tool_projection,
            )

            debug_boxes = candidates

        else:
            selected = self._select_highest_confidence(candidates)
            debug_boxes = [selected]

        selected["selected"] = True
        self._last_yolo_debug_boxes = debug_boxes

        self.node.get_logger().debug(
            f"YOLO {target_class_name} detection: "
            f"candidates={len(candidates)}, "
            f"selected conf={selected['conf']:.3f}, "
            f"center=({selected['center_u']:.1f}, {selected['center_v']:.1f}), "
            f"radius={selected['radius_px']:.1f}, "
            f"area={selected['area']:.1f}"
        )

        return (
            float(selected["center_u"]),
            float(selected["center_v"]),
            float(selected["radius_px"]),
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


    def draw_servo_overlay(
        self,
        image,
        target_u,
        target_v,
        radius_px,
        target_base,
        tool_pose,
        raw_error_x_base,
        raw_error_y_base,
        error_tool,
        depth_m,
        cmd,
        status,
    ):
        target_px = (int(round(target_u)), int(round(target_v)))

        # Draw YOLO target
        cv2.circle(image, target_px, int(round(radius_px)), (0, 255, 0), 2)
        cv2.drawMarker(
            image,
            target_px,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )

        # Draw projected tool0 only for visualization/debug.
        # Do not use this for control.
        tool_projection = self.node.motion.project_tool0_to_image()
        if tool_projection is not None:
            tool0_u, tool0_v, _tool0_z_cam = tool_projection
            tool_px = (int(round(tool0_u)), int(round(tool0_v)))

            cv2.drawMarker(
                image,
                tool_px,
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )

            cv2.arrowedLine(
                image,
                tool_px,
                target_px,
                (0, 255, 255),
                2,
                tipLength=0.2,
            )

            pixel_du = target_u - tool0_u
            pixel_dv = target_v - tool0_v
            tool_px_text = f"tool0_px: u={tool0_u:.1f}, v={tool0_v:.1f}"
            pixel_error_text = (
                f"pixel_error visual only: du={pixel_du:+.1f}, "
                f"dv={pixel_dv:+.1f}"
            )
        else:
            tool_px_text = "tool0_px: unavailable"
            pixel_error_text = "pixel_error visual only: unavailable"

        if target_base is not None:
            target_base_text = (
                f"target_base: x={target_base.point.x:+.4f}, "
                f"y={target_base.point.y:+.4f}, "
                f"z={target_base.point.z:+.4f} m"
            )
        else:
            target_base_text = "target_base: unavailable"

        if tool_pose is not None:
            tool_base_text = (
                f"tool0_base: x={tool_pose.pose.position.x:+.4f}, "
                f"y={tool_pose.pose.position.y:+.4f}, "
                f"z={tool_pose.pose.position.z:+.4f} m"
            )
        else:
            tool_base_text = "tool0_base: unavailable"

        if error_tool is not None:
            ex_tool, ey_tool, ez_tool = error_tool
            tool_error_text = (
                f"tool_error: x={ex_tool:+.4f}, "
                f"y={ey_tool:+.4f}, "
                f"z={ez_tool:+.4f} m"
            )
        else:
            tool_error_text = "tool_error: unavailable"

        if cmd is not None:
            vx, vy, vz = cmd
            cmd_text = (
                f"cmd tool0: vx={vx:+.4f}, "
                f"vy={vy:+.4f}, "
                f"vz={vz:+.4f} m/s"
            )
        else:
            cmd_text = "cmd tool0: zero"

        lines = [
            f"status: {status}",
            f"target_px: u={target_u:.1f}, v={target_v:.1f}, r={radius_px:.1f}",
            tool_px_text,
            pixel_error_text,
            target_base_text,
            tool_base_text,
            (
                f"base_error_xy: x={raw_error_x_base:+.4f}, "
                f"y={raw_error_y_base:+.4f} m"
            ),
            tool_error_text,
            f"depth: {depth_m:.4f} m",
            cmd_text,
        ]

        self.draw_text_block(image, lines, 10, 24)

    def draw_no_target_overlay(self, image):
        center_px = (int(round(self.cx())), int(round(self.cy())))
        cv2.drawMarker(image, center_px, (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        self.draw_text_block(image, ["status: no target", "cmd tool0: zero"], 10, 24)


    def draw_text_block(self, image, lines, x, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        line_height = 22
        for i, line in enumerate(lines):
            yy = y + i * line_height
            cv2.putText(image, line, (x + 1, yy + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(image, line, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


    def publish_debug_image(self, image, original_header):
        try:
            debug_msg = self.node.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            debug_msg.header = original_header
            self.node.debug_image_pub.publish(debug_msg)
        except Exception as exc:
            self.node.get_logger().warn(f"Failed to publish debug image: {exc}")

    def fx(self):
        return float(self.node.get_parameter("fx").value)

    def fy(self):
        return float(self.node.get_parameter("fy").value)

    def cx(self):
        return float(self.node.get_parameter("cx").value)

    def cy(self):
        return float(self.node.get_parameter("cy").value)