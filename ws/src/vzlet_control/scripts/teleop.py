#!/usr/bin/env python3

import argparse
import csv
import select
import sys
import termios
import threading
import time
import tty
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from controller_manager_msgs.srv import SwitchController
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TwistStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import ServoCommandType
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener


CSV_HEADERS = ["name", "id", "x", "y", "z", "qx", "qy", "qz", "qw"]


def normalize_row(row):
    return {
        (key or "").strip(): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
        if key is not None
    }


def read_rows(path: Path):
    if not path.exists():
        return [], []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [normalize_row(row) for row in reader]

    return fieldnames, rows


def ensure_trailing_newline(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("rb") as f:
        f.seek(-1, 2)
        last_char = f.read(1)

    if last_char != b"\n":
        with path.open("a", newline="") as f:
            f.write("\n")


def ensure_csv_exists(path: Path):
    if path.exists():
        ensure_trailing_newline(path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()


class TrajectoryTeleopCommander(Node):

    def __init__(self, csv_file: str):
        super().__init__("trajectory_teleop_commander")

        self.declare_parameter("base_frame", "world")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("frame_id", "world")

        self.declare_parameter("move_action", "/move_action")
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("planner_id", "RRTConnect")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("velocity_scaling", 0.2)
        self.declare_parameter("acceleration_scaling", 0.2)
        self.declare_parameter("position_tolerance", 0.001)
        self.declare_parameter("orientation_tolerance", 0.01)

        self.declare_parameter("servo_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("linear_speed", 0.4)
        self.declare_parameter("publish_rate", 30.0)

        self.declare_parameter(
            "controller_switch_service",
            "/controller_manager/switch_controller",
        )
        self.declare_parameter("trajectory_controller", "joint_trajectory_controller")
        self.declare_parameter("servo_controller", "forward_position_controller")
        self.declare_parameter("controller_switch_timeout", 5.0)

        self.declare_parameter(
            "servo_command_type_service",
            "/servo_node/switch_command_type",
        )
        self.declare_parameter("servo_command_type", 1)

        self.declare_parameter("gripper_action", "/gripper_controller/gripper_cmd")
        self.declare_parameter("gripper_open_position", 0.5)
        self.declare_parameter("gripper_close_position", 0.04)

        # Dataset photo recording parameters.
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("save_every_n", 45)
        self.declare_parameter("output_dir", "dataset_output")
        self.declare_parameter("image_extension", "jpg")

        self.csv_file = Path(csv_file)
        ensure_csv_exists(self.csv_file)
        self.poses = self.load_poses()

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.servo_topic = str(self.get_parameter("servo_topic").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)

        self.trajectory_controller = str(
            self.get_parameter("trajectory_controller").value
        )
        self.servo_controller = str(
            self.get_parameter("servo_controller").value
        )

        self.gripper_action = str(self.get_parameter("gripper_action").value)
        self.gripper_open_position = float(
            self.get_parameter("gripper_open_position").value
        )
        self.gripper_close_position = float(
            self.get_parameter("gripper_close_position").value
        )

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.save_every_n = int(self.get_parameter("save_every_n").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.image_extension = str(
            self.get_parameter("image_extension").value
        ).lower()

        if self.save_every_n < 1:
            self.get_logger().warn("save_every_n must be >= 1. Falling back to 1.")
            self.save_every_n = 1

        if self.image_extension not in ["jpg", "jpeg", "png"]:
            self.get_logger().warn("Unsupported image_extension. Falling back to jpg.")
            self.image_extension = "jpg"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.in_teleop = False

        self.gripper_is_closed = False
        self.gripper_busy = False

        self.last_record_time_s = 0.0
        self.record_debounce_s = 0.5

        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id
        self.lock = threading.Lock()

        self.bridge = CvBridge()
        self.dataset_lock = threading.Lock()
        self.dataset_recording = False
        self.dataset_dir = None
        self.zip_path = None
        self.image_count = 0
        self.saved_count = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.servo_pub = self.create_publisher(
            TwistStamped,
            self.servo_topic,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.movegroup_client = ActionClient(
            self,
            MoveGroup,
            str(self.get_parameter("move_action").value),
        )

        self.gripper_client = ActionClient(
            self,
            ParallelGripperCommand,
            self.gripper_action,
        )

        self.switch_controller_client = self.create_client(
            SwitchController,
            str(self.get_parameter("controller_switch_service").value),
        )

        self.servo_command_type_client = self.create_client(
            ServoCommandType,
            str(self.get_parameter("servo_command_type_service").value),
        )

        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_servo_command,
        )

        self.get_logger().info(f"Loaded {len(self.poses)} poses from: {self.csv_file}")
        self.get_logger().info(f"Servo topic: {self.servo_topic}")
        self.get_logger().info(f"Frame ID: {self.frame_id}")
        self.get_logger().info(f"Gripper action: {self.gripper_action}")
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(
            f"Dataset hotkey: z toggles photo recording, save_every_n={self.save_every_n}"
        )
        self.get_logger().info("Pose hotkey: p records current EEF pose")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def wait_future(
        self,
        future,
        timeout_s: Optional[float],
        label: str,
        allow_hotkeys: bool = False,
    ) -> bool:
        start_s = time.monotonic()

        while rclpy.ok() and not future.done():
            if allow_hotkeys:
                self.process_pending_global_hotkeys()

            if timeout_s is not None and (time.monotonic() - start_s) > timeout_s:
                self.get_logger().error(f"{label}: timed out")
                return False

            time.sleep(0.01)

        return future.done()

    def enable_keyboard_hotkeys(self):
        if not sys.stdin.isatty():
            return None

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return old_settings

    def restore_keyboard_hotkeys(self, old_settings):
        if old_settings is None:
            return

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def process_pending_global_hotkeys(self):
        if not sys.stdin.isatty():
            return

        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            self.handle_global_hotkey(key)

    def handle_global_hotkey(self, key: str) -> bool:
        if key == "z":
            self.toggle_dataset_recording()
            return True

        if key == "p":
            self.record_current_pose()
            return True

        if key == "\x03":
            self.running = False
            raise KeyboardInterrupt

        return False

    def make_unique_dataset_paths(self):
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        dataset_dir = self.output_dir / f"ds_{timestamp}"
        zip_path = self.output_dir / f"ds_{timestamp}.zip"

        suffix = 1
        while dataset_dir.exists() or zip_path.exists():
            dataset_dir = self.output_dir / f"ds_{timestamp}_{suffix}"
            zip_path = self.output_dir / f"ds_{timestamp}_{suffix}.zip"
            suffix += 1

        return dataset_dir, zip_path

    def start_dataset_recording(self):
        with self.dataset_lock:
            if self.dataset_recording:
                self.get_logger().warn("Dataset photo recording is already running")
                return

            self.dataset_dir, self.zip_path = self.make_unique_dataset_paths()
            self.dataset_dir.mkdir(parents=True, exist_ok=False)

            self.image_count = 0
            self.saved_count = 0
            self.dataset_recording = True

            self.get_logger().info("Dataset photo recording started")
            self.get_logger().info(f"Recording every {self.save_every_n}th image")
            self.get_logger().info(f"Saving dataset to: {self.dataset_dir}")

    def stop_dataset_recording(self):
        with self.dataset_lock:
            if not self.dataset_recording:
                self.get_logger().warn("Dataset photo recording is not running")
                return

            self.dataset_recording = False
            dataset_dir = self.dataset_dir
            zip_path = self.zip_path
            saved_count = self.saved_count

        self.create_zip_archive(dataset_dir, zip_path, saved_count)

    def toggle_dataset_recording(self):
        with self.dataset_lock:
            is_recording = self.dataset_recording

        if is_recording:
            self.stop_dataset_recording()
        else:
            self.start_dataset_recording()

    def image_callback(self, msg: Image):
        with self.dataset_lock:
            if not self.dataset_recording:
                return

            self.image_count += 1

            if self.image_count % self.save_every_n != 0:
                return

            dataset_dir = self.dataset_dir
            image_extension = self.image_extension
            image_index = self.saved_count

            try:
                cv_image = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )

                filename = f"img_{image_index:06d}.{image_extension}"
                filepath = dataset_dir / filename

                if image_extension in ["jpg", "jpeg"]:
                    ok = cv2.imwrite(
                        str(filepath),
                        cv_image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                else:
                    ok = cv2.imwrite(str(filepath), cv_image)

                if not ok:
                    self.get_logger().error(f"Failed to write image: {filepath}")
                    return

                self.saved_count += 1

                if self.saved_count % 50 == 0:
                    self.get_logger().info(f"Saved {self.saved_count} images")

            except Exception as exc:
                self.get_logger().error(f"Failed to save image: {exc}")

    def create_zip_archive(self, dataset_dir, zip_path, saved_count):
        if dataset_dir is None or zip_path is None:
            self.get_logger().warn("No dataset directory exists, zip archive not created")
            return

        if not dataset_dir.exists():
            self.get_logger().warn(
                f"Dataset directory does not exist, zip archive not created: {dataset_dir}"
            )
            return

        self.get_logger().info("Creating zip archive...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dataset_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_dir.parent)
                    zipf.write(file_path, arcname)

        self.get_logger().info(f"Zip archive created: {zip_path}")
        self.get_logger().info(f"Total saved images: {saved_count}")

    def load_poses(self) -> list:
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file does not exist: {self.csv_file}")

        poses = []

        with self.csv_file.open("r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                normalized = normalize_row(row)

                if not normalized:
                    continue

                pose = {
                    "name": normalized.get("name", ""),
                    "id": int(normalized["id"]),
                    "x": float(normalized["x"]),
                    "y": float(normalized["y"]),
                    "z": float(normalized["z"]),
                    "qx": float(normalized["qx"]),
                    "qy": float(normalized["qy"]),
                    "qz": float(normalized["qz"]),
                    "qw": float(normalized["qw"]),
                }

                poses.append(pose)

        return poses

    def reload_poses(self):
        self.poses = self.load_poses()
        self.get_logger().info(f"Reloaded {len(self.poses)} poses from CSV")

    def get_next_counter(self):
        self.reload_poses()
        return len(self.poses) + 1

    def print_main_menu(self):
        print(
            """
Main menu:

  l             -> list parsed CSV poses
  t             -> enter teleop mode
  z             -> start/stop dataset photo recording
  p             -> record current tool0 pose to CSV
  <pose_name>   -> switch to joint_trajectory_controller and move to CSV pose
  q / quit      -> exit

During trajectory execution:
  z             -> start/stop dataset photo recording
  p             -> record current tool0 pose to CSV
  Ctrl+C        -> exit

Examples:
  zone1
  00
  12_pre
"""
        )

    def print_teleop_controls(self):
        print(
            """
Teleop mode:

  w / s  -> +X / -X
  a / d  -> +Y / -Y
  r / f  -> +Z / -Z

  g      -> toggle gripper open/close
  z      -> start/stop dataset photo recording
  p      -> record current tool0 pose to CSV
  t      -> exit teleop and return to main menu
  Ctrl+C -> exit
"""
        )

    def print_available_poses(self):
        self.reload_poses()

        print("\nAvailable poses:")
        for pose in self.poses:
            print(
                f"  {pose['name']} -> "
                f"x={pose['x']:.4f}, y={pose['y']:.4f}, z={pose['z']:.4f}"
            )
        print("")

    def find_pose(self, command: str) -> Optional[dict]:
        pose_name = command.strip()

        if not pose_name:
            return None

        for pose in self.poses:
            if str(pose["name"]).strip() == pose_name:
                return pose

        return None

    def call_switch_controller(
        self,
        activate_controllers=None,
        deactivate_controllers=None,
        label="switch controllers",
    ) -> bool:
        activate_controllers = activate_controllers or []
        deactivate_controllers = deactivate_controllers or []

        if not self.switch_controller_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Controller switch service is not available")
            return False

        request = SwitchController.Request()
        request.activate_controllers = list(activate_controllers)
        request.deactivate_controllers = list(deactivate_controllers)

        # BEST_EFFORT:
        # Do not fail if a controller is already active/inactive.
        if hasattr(request, "BEST_EFFORT"):
            request.strictness = request.BEST_EFFORT
        else:
            request.strictness = 1

        if hasattr(request, "activate_asap"):
            request.activate_asap = True

        timeout_s = float(self.get_parameter("controller_switch_timeout").value)

        if hasattr(request, "timeout"):
            request.timeout.sec = int(timeout_s)
            request.timeout.nanosec = int((timeout_s - int(timeout_s)) * 1e9)

        self.get_logger().info(
            f"{label}: activate={activate_controllers}, "
            f"deactivate={deactivate_controllers}, strictness=BEST_EFFORT"
        )

        future = self.switch_controller_client.call_async(request)

        if not self.wait_future(future, timeout_s + 5.0, label):
            return False

        response = future.result()

        if response is None:
            self.get_logger().error(f"{label}: no response")
            return False

        if hasattr(response, "ok") and not response.ok:
            self.get_logger().error(f"{label}: rejected by controller manager")
            return False

        self.get_logger().info(f"{label}: succeeded")
        return True

    def switch_to_trajectory_mode(self) -> bool:
        return self.call_switch_controller(
            deactivate_controllers=[self.servo_controller],
            activate_controllers=[self.trajectory_controller],
            label="switch to trajectory controller mode",
        )

    def switch_to_teleop_mode(self) -> bool:
        if not self.call_switch_controller(
            deactivate_controllers=[self.trajectory_controller],
            activate_controllers=[self.servo_controller],
            label="switch to teleop controller mode",
        ):
            return False

        if not self.set_servo_command_type():
            return False

        return True

    def set_servo_command_type(self) -> bool:
        if not self.servo_command_type_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("Servo command-type service is not available")
            return False

        command_type = int(self.get_parameter("servo_command_type").value)

        request = ServoCommandType.Request()
        request.command_type = command_type

        self.get_logger().info(
            f"Setting MoveIt Servo command type: command_type={command_type}"
        )

        future = self.servo_command_type_client.call_async(request)

        if not self.wait_future(future, 5.0, "set Servo command type"):
            return False

        response = future.result()

        if response is None:
            self.get_logger().error("Servo command-type service returned no response")
            return False

        if hasattr(response, "success") and not response.success:
            self.get_logger().error("Servo command-type service rejected request")
            return False

        self.get_logger().info("Servo command type set")
        return True

    def pose_to_pose_stamped(self, pose: dict) -> PoseStamped:
        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.header.stamp = self.get_clock().now().to_msg()

        target.pose.position.x = pose["x"]
        target.pose.position.y = pose["y"]
        target.pose.position.z = pose["z"]

        target.pose.orientation.x = pose["qx"]
        target.pose.orientation.y = pose["qy"]
        target.pose.orientation.z = pose["qz"]
        target.pose.orientation.w = pose["qw"]

        return target

    def create_move_goal(self, target_pose: PoseStamped) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        request = MotionPlanRequest()

        request.group_name = self.get_parameter("group_name").value
        request.pipeline_id = str(self.get_parameter("pipeline_id").value)
        request.planner_id = str(self.get_parameter("planner_id").value)
        request.num_planning_attempts = int(
            self.get_parameter("planning_attempts").value
        )
        request.allowed_planning_time = float(
            self.get_parameter("allowed_planning_time").value
        )
        request.max_velocity_scaling_factor = float(
            self.get_parameter("velocity_scaling").value
        )
        request.max_acceleration_scaling_factor = float(
            self.get_parameter("acceleration_scaling").value
        )

        constraints = Constraints()

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.base_frame
        position_constraint.link_name = self.tool_frame

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [
            float(self.get_parameter("position_tolerance").value)
        ]

        volume = BoundingVolume()
        volume.primitives.append(primitive)
        volume.primitive_poses.append(target_pose.pose)

        position_constraint.constraint_region = volume
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.base_frame
        orientation_constraint.link_name = self.tool_frame
        orientation_constraint.orientation = target_pose.pose.orientation

        orientation_tolerance = float(
            self.get_parameter("orientation_tolerance").value
        )
        orientation_constraint.absolute_x_axis_tolerance = orientation_tolerance
        orientation_constraint.absolute_y_axis_tolerance = orientation_tolerance
        orientation_constraint.absolute_z_axis_tolerance = orientation_tolerance
        orientation_constraint.weight = 1.0

        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)

        request.goal_constraints.append(constraints)

        goal.request = request
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        return goal

    def execute_pose(self, pose: dict) -> bool:
        target_pose = self.pose_to_pose_stamped(pose)

        self.get_logger().info(
            f"Executing pose {pose['name']}: "
            f"x={pose['x']:.4f}, y={pose['y']:.4f}, z={pose['z']:.4f}, "
            f"qx={pose['qx']:.4f}, qy={pose['qy']:.4f}, "
            f"qz={pose['qz']:.4f}, qw={pose['qw']:.4f}"
        )

        self.get_logger().info("Waiting for MoveGroup action server...")

        if not self.movegroup_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup action server is not available")
            return False

        self.get_logger().info("MoveGroup connected")
        self.get_logger().info("During execution: press z to toggle image recording, p to record pose")

        goal = self.create_move_goal(target_pose)
        send_goal_future = self.movegroup_client.send_goal_async(goal)

        old_settings = self.enable_keyboard_hotkeys()

        try:
            if not self.wait_future(
                send_goal_future,
                30.0,
                "wait for MoveGroup goal acceptance",
                allow_hotkeys=True,
            ):
                return False

            goal_handle = send_goal_future.result()

            if goal_handle is None:
                self.get_logger().error("Goal handle is None")
                return False

            if not goal_handle.accepted:
                self.get_logger().error("Goal rejected")
                return False

            self.get_logger().info("Goal accepted, waiting for execution to finish...")

            result_future = goal_handle.get_result_async()

            if not self.wait_future(
                result_future,
                300.0,
                "wait for MoveGroup execution result",
                allow_hotkeys=True,
            ):
                return False

            result = result_future.result()

            if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"Pose {pose['name']} executed successfully")
                return True

            status = None if result is None else result.status
            self.get_logger().error(f"Pose execution failed with status: {status}")
            return False

        finally:
            self.restore_keyboard_hotkeys(old_settings)

    def publish_servo_command(self):
        with self.lock:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            msg.twist.linear.x = self.current_twist.twist.linear.x
            msg.twist.linear.y = self.current_twist.twist.linear.y
            msg.twist.linear.z = self.current_twist.twist.linear.z

        self.servo_pub.publish(msg)

    def publish_stop(self):
        self.set_velocity()

        for _ in range(5):
            self.publish_servo_command()
            time.sleep(0.02)

    def set_velocity(self, x=0.0, y=0.0, z=0.0):
        with self.lock:
            self.current_twist.twist.linear.x = x
            self.current_twist.twist.linear.y = y
            self.current_twist.twist.linear.z = z

    def toggle_gripper(self):
        if self.gripper_busy:
            self.get_logger().warn("Gripper command already in progress")
            return

        if not self.gripper_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn("Gripper action server is not available")
            return

        target_position = (
            self.gripper_open_position
            if self.gripper_is_closed
            else self.gripper_close_position
        )
        label = "open" if self.gripper_is_closed else "close"

        goal = ParallelGripperCommand.Goal()
        goal.command.position = [target_position]

        self.gripper_busy = True
        self.get_logger().info(
            f"Gripper {label}: position={target_position:.4f}"
        )

        send_goal_future = self.gripper_client.send_goal_async(goal)
        send_goal_future.add_done_callback(
            lambda future: self._gripper_goal_response_callback(future, label)
        )

    def _gripper_goal_response_callback(self, future, label):
        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error(f"Gripper {label}: goal handle is None")
            self.gripper_busy = False
            return

        if not goal_handle.accepted:
            self.get_logger().error(f"Gripper {label}: goal rejected")
            self.gripper_busy = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self._gripper_result_callback(future, label)
        )

    def _gripper_result_callback(self, future, label):
        result = future.result()

        if result is None:
            self.get_logger().error(f"Gripper {label}: no result")
            self.gripper_busy = False
            return

        self.gripper_is_closed = label == "close"
        self.gripper_busy = False
        self.get_logger().info(f"Gripper {label}: done")

    def record_current_pose(self):
        now_s = self.now_s()

        if (now_s - self.last_record_time_s) < self.record_debounce_s:
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
                "id": counter,
                "x": t.x,
                "y": t.y,
                "z": t.z,
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "qw": q.w,
            }

            ensure_trailing_newline(self.csv_file)

            with self.csv_file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(row)

            self.reload_poses()

            self.get_logger().info(
                f"Recorded pose {row['name']} (#{counter}): "
                f"x={t.x:.6f}, y={t.y:.6f}, z={t.z:.6f}, "
                f"qx={q.x:.6f}, qy={q.y:.6f}, qz={q.z:.6f}, qw={q.w:.6f}"
            )

        except Exception as exc:
            self.get_logger().warn(f"TF lookup failed, pose not recorded: {exc}")

    def enter_teleop_mode(self):
        self.get_logger().info("Switching to teleop controller mode...")

        if not self.switch_to_teleop_mode():
            self.get_logger().error("Could not switch to teleop mode")
            return

        self.in_teleop = True
        self.publish_stop()
        self.print_teleop_controls()

        old_settings = self.enable_keyboard_hotkeys()

        try:
            while self.running and self.in_teleop and rclpy.ok():
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)

                    speed = self.linear_speed

                    if key == "w":
                        self.set_velocity(y=speed)
                    elif key == "s":
                        self.set_velocity(y=-speed)
                    elif key == "a":
                        self.set_velocity(x=-speed)
                    elif key == "d":
                        self.set_velocity(x=speed)
                    elif key == "r":
                        self.set_velocity(z=speed)
                    elif key == "f":
                        self.set_velocity(z=-speed)
                    elif key == "g":
                        self.publish_stop()
                        self.toggle_gripper()
                    elif key == "z":
                        self.toggle_dataset_recording()
                    elif key == "p":
                        self.publish_stop()
                        self.record_current_pose()
                    elif key == "t":
                        self.publish_stop()
                        self.in_teleop = False
                        self.get_logger().info("Leaving teleop mode")
                    elif key == "\x03":
                        raise KeyboardInterrupt
                    else:
                        self.set_velocity()

        except KeyboardInterrupt:
            self.running = False
            raise

        finally:
            self.publish_stop()
            self.restore_keyboard_hotkeys(old_settings)

    def interactive_loop(self):
        self.print_main_menu()

        while self.running and rclpy.ok():
            try:
                command = input("main> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                return

            if not command:
                continue

            if command in ("q", "quit", "exit"):
                self.running = False
                return

            if command == "l":
                self.print_available_poses()
                continue

            if command == "z":
                self.toggle_dataset_recording()
                continue

            if command == "p":
                self.record_current_pose()
                continue

            if command == "t":
                self.enter_teleop_mode()
                self.print_main_menu()
                continue

            self.reload_poses()
            pose = self.find_pose(command)

            if pose is None:
                self.get_logger().warn(
                    f"Unknown command or pose name: {command}. "
                    f"Use 'l' to list poses, 't' for teleop, 'z' for images, "
                    f"'p' for pose, or 'q' to quit."
                )
                continue

            self.get_logger().info(
                "Switching to trajectory controller mode before pose execution..."
            )

            if not self.switch_to_trajectory_mode():
                self.get_logger().error("Could not switch to trajectory controller mode")
                continue

            self.execute_pose(pose)

    def stop(self):
        self.running = False
        self.in_teleop = False
        self.publish_stop()

        with self.dataset_lock:
            should_stop_dataset = self.dataset_recording

        if should_stop_dataset:
            self.stop_dataset_recording()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Trajectory MoveIt pose commander, keyboard teleop, pose recorder, and image dataset recorder."
    )

    parser.add_argument(
        "--csv",
        default="/home/sonieth2/vzlet_ur3e/ws/zone_poses_floor.csv",
        help="CSV file with EEF poses.",
    )

    return parser.parse_known_args(argv)


def main(argv=None):
    cli_args, ros_args = parse_args(argv)

    rclpy.init(args=ros_args)

    node = TrajectoryTeleopCommander(cli_args.csv)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )
    spin_thread.start()

    try:
        node.interactive_loop()

    except KeyboardInterrupt:
        node.get_logger().info("Shutting down trajectory teleop commander...")

    finally:
        node.stop()

        executor.shutdown()
        spin_thread.join(timeout=1.0)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
