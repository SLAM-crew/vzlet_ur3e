import time
import rclpy
from controller_manager_msgs.srv import SwitchController, ListControllers
from moveit_msgs.srv import ServoCommandType
import csv
from pathlib import Path
from typing import Optional
from geometry_msgs.msg import PoseStamped

CSV_HEADERS = ["name", "id", "x", "y", "z", "qx", "qy", "qz", "qw"]

class PipelineUtils:
    def __init__(self, node):
        self.node = node
        self.poses: dict[str, dict] = {}

        switch_service = str(
            self.node.get_parameter("controller_switch_service").value
        )

        list_service = switch_service.replace(
            "/switch_controller",
            "/list_controllers",
        )

        self.switch_controller_client = self.node.create_client(
            SwitchController,
            switch_service,
        )

        self.list_controllers_client = self.node.create_client(
            ListControllers,
            list_service,
        )

        self.servo_command_type_client = self.node.create_client(
            ServoCommandType,
            str(self.node.get_parameter("servo_command_type_service").value),
        )

    def load_csv_poses(self, csv_file: Optional[str] = None) -> dict[str, dict]:
        if csv_file is None:
            csv_file = self.node.get_parameter("zone_pose_csv").value

        csv_path = Path(csv_file)

        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        poses = {}
        required_columns = ["name", "id", "x", "y", "z", "qx", "qy", "qz", "qw"]

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(f"CSV file is empty: {csv_file}")

            fieldnames = [(field or "").strip() for field in reader.fieldnames]
            missing_columns = [
                column for column in required_columns
                if column not in fieldnames
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required CSV columns: {missing_columns}"
                )

            for row_index, row in enumerate(reader, start=2):
                normalized = {
                    (key or "").strip(): (value.strip() if isinstance(value, str) else value)
                    for key, value in row.items()
                    if key is not None
                }

                name = normalized["name"].strip()

                if not name:
                    raise ValueError(f"Empty pose name at CSV row {row_index}")

                if name in poses:
                    raise ValueError(
                        f"Duplicate pose name '{name}' at CSV row {row_index}"
                    )

                try:
                    poses[name] = {
                        "id": int(normalized["id"]),
                        "x": float(normalized["x"]),
                        "y": float(normalized["y"]),
                        "z": float(normalized["z"]),
                        "qx": float(normalized["qx"]),
                        "qy": float(normalized["qy"]),
                        "qz": float(normalized["qz"]),
                        "qw": float(normalized["qw"]),
                    }
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid numeric value at CSV row {row_index}, pose '{name}': {exc}"
                    )

        if not poses:
            raise ValueError(f"No poses found in CSV: {csv_file}")

        self.poses = poses

        self.node.get_logger().info(
            f"Loaded {len(self.poses)} poses from CSV: {csv_file}"
        )

        return self.poses

    def get_pose_by_name(
        self,
        pose_name: str,
        log_level: str = "error",
    ) -> Optional[dict]:
        pose_name = str(pose_name).strip()

        if not pose_name:
            self._log(log_level, f"Requested name dont exist: '{pose_name}'")
            return None

        try:
            return self.poses[pose_name]
        except KeyError:
            available_names = ", ".join(sorted(self.poses.keys()))
            self._log(
                log_level,
                f"'{pose_name}' was not found in loaded CSV; "
                f"available names: {available_names}",
            )
            return None

    def get_pose_by_id(
        self,
        command: str,
        log_level: str = "error",
    ) -> Optional[dict]:
        pose_id_text = str(command).strip()

        if not pose_id_text.isdigit():
            self._log(
                log_level,
                f"Pose command must be a numeric CSV id, got: {command}",
            )
            return None

        requested_id = int(pose_id_text)

        for pose in self.poses.values():
            try:
                pose_id = int(str(pose.get("id", "")).strip())
            except Exception:
                continue

            if pose_id == requested_id:
                return pose

        available_ids = []
        for pose in self.poses.values():
            pose_id = str(pose.get("id", "")).strip()
            if pose_id:
                available_ids.append(pose_id)

        self._log(
            log_level,
            f"Pose id {requested_id} was not found in loaded CSV; "
            f"available ids: {', '.join(sorted(available_ids, key=lambda item: int(item) if item.isdigit() else 999999))}",
        )
        return None
    
    def get_name_by_id(self, pose_id, log_level: str = "error") -> Optional[str]:
        try:
            requested_id = int(str(pose_id).strip())
        except ValueError:
            self._log(log_level, f"Pose id must be numeric, got: {pose_id}")
            return None

        for name, pose in self.poses.items():
            if int(pose.get("id", -1)) == requested_id:
                return name

        available_ids = sorted(
            str(pose["id"]) for pose in self.poses.values() if "id" in pose
        )

        self._log(
            log_level,
            f"Pose id {requested_id} was not found in loaded CSV; "
            f"available ids: {', '.join(available_ids)}",
        )
        return None
    
    def _log(self, log_level: str, message: str):
        logger = self.node.get_logger()

        if log_level == "warn":
            logger.warn(message)
        elif log_level == "info":
            logger.info(message)
        else:
            logger.error(message)

    def save_pose_to_csv(self):
        now_s = self.now_s()
        # self.record_debounce_s
        if (now_s - self.last_record_time_s) < 0.5:
            self.get_logger().warn("Record ignored: debounce active")
            return

        self.last_record_time_s = now_s

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            counter = self.get_next_counter()

            row = {
                "name": f"zone{counter}",
                "id": str(counter),
                "x": t.x,
                "y": t.y,
                "z": t.z,
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "qw": q.w,
            }

            self.utils.ensure_trailing_newline(self.csv_file)

            with self.csv_file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(row)

            self.reload_poses()

            self.get_logger().info(
                f"Recorded pose {row['name']}: "
                f"x={t.x:.6f}, y={t.y:.6f}, z={t.z:.6f}, "
                f"qx={q.x:.6f}, qy={q.y:.6f}, qz={q.z:.6f}, qw={q.w:.6f}"
            )

        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed, pose not recorded: {exc}")

    
    def ensure_trailing_newline(path: Path):
        if not path.exists() or path.stat().st_size == 0:
            return

        with path.open("rb") as f:
            f.seek(-1, 2)
            last_char = f.read(1)

        if last_char != b"\n":
            with path.open("a", newline="") as f:
                f.write("\n")

    def pose_to_pose_stamped(self, pose: dict) -> PoseStamped:
        target = PoseStamped()
        target.header.frame_id = self.node.base_frame
        target.header.stamp = self.node.get_clock().now().to_msg()

        target.pose.position.x = pose["x"]
        target.pose.position.y = pose["y"]
        target.pose.position.z = pose["z"]

        target.pose.orientation.x = pose["qx"]
        target.pose.orientation.y = pose["qy"]
        target.pose.orientation.z = pose["qz"]
        target.pose.orientation.w = pose["qw"]

        return target
    
    ########################################################

    def wait_future(self, future, timeout_s, label: str, tick_fn=None) -> bool:
        use_background_executor = bool(
            getattr(self.node, "use_background_executor", False)
        )

        start_s = time.monotonic()

        while rclpy.ok() and not future.done():
            if tick_fn is not None:
                tick_fn()

            if timeout_s is not None and (time.monotonic() - start_s) > timeout_s:
                self.node.get_logger().error(f"{label}: timed out")
                return False

            if use_background_executor:
                time.sleep(0.01)
            else:
                rclpy.spin_once(self.node, timeout_sec=0.01)

        return future.done()


    def call_switch_controller(
            self,
            activate_controllers=None,
            deactivate_controllers=None,
            label="switch controllers",
        ) -> bool:
        activate_controllers = activate_controllers or []
        deactivate_controllers = deactivate_controllers or []

        if self.controllers_already_in_state(
            activate_controllers,
            deactivate_controllers,
        ):
            return True

        if not self.switch_controller_client.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().error("Controller switch service is not available")
            return False

        request = SwitchController.Request()
        request.activate_controllers = list(activate_controllers)
        request.deactivate_controllers = list(deactivate_controllers)

        if hasattr(request, "BEST_EFFORT"):
            request.strictness = request.BEST_EFFORT
        else:
            request.strictness = 1

        if hasattr(request, "activate_asap"):
            request.activate_asap = True

        timeout_s = float(
            self.node.get_parameter("controller_switch_timeout").value
        )

        if hasattr(request, "timeout"):
            request.timeout.sec = int(timeout_s)
            request.timeout.nanosec = int((timeout_s - int(timeout_s)) * 1e9)

        self.node.get_logger().info(
            f"{label}: activate={activate_controllers}, "
            f"deactivate={deactivate_controllers}, strictness=BEST_EFFORT"
        )

        future = self.switch_controller_client.call_async(request)

        if not self.wait_future(future, timeout_s + 5.0, label):
            return False

        response = future.result()

        if response is None:
            self.node.get_logger().error(f"{label}: no response")
            return False

        if hasattr(response, "ok") and not response.ok:
            # Handle harmless no-op / race case.
            if self.controllers_already_in_state(
                activate_controllers,
                deactivate_controllers,
            ):
                self.node.get_logger().warn(
                    f"{label}: switch reported failure, but controller state is correct"
                )
                return True

            self.node.get_logger().error(f"{label}: rejected by controller manager")
            return False

        self.node.get_logger().info(f"{label}: succeeded")
        return True

    def switch_to_trajectory_mode(self) -> bool:
        trajectory_controller = str(
            self.node.get_parameter("trajectory_controller").value
        )
        servo_controller = str(
            self.node.get_parameter("servo_controller").value
        )

        return self.call_switch_controller(
            deactivate_controllers=[servo_controller],
            activate_controllers=[trajectory_controller],
            label="switch to trajectory controller mode",
        )


    def switch_to_teleop_mode(self) -> bool:
        trajectory_controller = str(
            self.node.get_parameter("trajectory_controller").value
        )
        servo_controller = str(
            self.node.get_parameter("servo_controller").value
        )

        if not self.call_switch_controller(
            deactivate_controllers=[trajectory_controller],
            activate_controllers=[servo_controller],
            label="switch to teleop controller mode",
        ):
            return False

        return self.set_servo_command_type()


    def set_servo_command_type(self) -> bool:
        if not self.servo_command_type_client.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().error("Servo command-type service is not available")
            return False

        command_type = int(self.node.get_parameter("servo_command_type").value)

        request = ServoCommandType.Request()
        request.command_type = command_type

        self.node.get_logger().info(
            f"Setting MoveIt Servo command type: command_type={command_type}"
        )

        future = self.servo_command_type_client.call_async(request)

        if not self.wait_future(future, 5.0, "set Servo command type"):
            return False

        response = future.result()

        if response is None:
            self.node.get_logger().error("Servo command-type service returned no response")
            return False

        if hasattr(response, "success") and not response.success:
            self.node.get_logger().error("Servo command-type service rejected request")
            return False

        self.node.get_logger().info("Servo command type set")
        return True
    
    def get_controller_states(self) -> Optional[dict]:
        self.node.get_logger().info("Checking /controller_manager/list_controllers service...")

        if not self.list_controllers_client.wait_for_service(timeout_sec=3.0):
            self.node.get_logger().error("Controller list service is not available")
            return None

        self.node.get_logger().info("Calling list_controllers...")

        future = self.list_controllers_client.call_async(
            ListControllers.Request()
        )

        if not self.wait_future(future, 5.0, "list controllers"):
            return None

        response = future.result()
        if response is None:
            self.node.get_logger().error("Controller list service returned no response")
            return None

        states = {
            controller.name: controller.state
            for controller in response.controller
        }

        # self.node.get_logger().info(f"Controller states: {states}")
        return states

    def controllers_already_in_state(
        self,
        activate_controllers,
        deactivate_controllers,
    ) -> bool:
        states = self.get_controller_states()

        if states is None:
            return False

        for controller in activate_controllers:
            if states.get(controller) != "active":
                return False

        for controller in deactivate_controllers:
            if states.get(controller) == "active":
                return False

        self.node.get_logger().info(
            "Controllers already in requested state: "
            f"active={activate_controllers}, inactive={deactivate_controllers}"
        )
        return True