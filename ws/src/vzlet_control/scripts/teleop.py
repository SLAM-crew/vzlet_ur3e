#!/home/sonieth2/vzlet_ur3e/ur_rtde_scripts/venv/bin/python3

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

import cv2
import rclpy
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import TwistStamped
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image

from base_node import BaseRobotNode
from pipeline_motion import MotionController
from pipeline_utils import PipelineUtils

CSV_HEADERS = ["name", "id", "x", "y", "z", "qx", "qy", "qz", "qw"]

class TrajectoryTeleopCommander(BaseRobotNode):

    def __init__(self, csv_file: str):
        super().__init__("traj_tele")

        self.use_background_executor = True
        self.csv_file = Path(csv_file)

        self.declare_parameter("linear_speed", 0.4)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("output_dir", "dataset_output")

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.utils = PipelineUtils(self)
        self.motion = MotionController(self)

        try:
            self.utils.load_csv_poses(str(self.csv_file))
        except Exception as exc:
            self.get_logger().error(f"Could not load zone poses: {exc}")

        self.selected_grasp_object = "body"

        self.running = True
        self.in_teleop = False

        self.gripper_is_closed = False
        self.gripper_busy = False

        self.last_record_time_s = 0.0

        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.base_frame
        self.lock = threading.Lock()

        self.dataset_lock = threading.Lock()
        self.dataset_dir = None
        self.zip_path = None
        self.saved_count = 0
        self.latest_image = None
        self.latest_image_stamp = None
        self.last_snapshot_time_s = 0.0
        self.dataset_archived = False

        self.servo_pub = self.create_publisher(
            TwistStamped,
            self.node.servo_topic,
            10,
        )
        
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_servo_command,
        )

        self.get_logger().info(f"Loaded {len(self.utils.poses)} poses from: {self.csv_file}")
        self.get_logger().info(f"Servo topic: {self.node.servo_topic}")
        self.get_logger().info(f"Frame ID: {self.base_frame}")
        self.get_logger().info(f"Gripper action: {self.gripper_action}")
        self.get_logger().info(
            f"Selected grasp object: {self.selected_grasp_object} "
        )
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(
            "Dataset hotkey: z captures one photo; archive is created on node shutdown"
        )
        self.get_logger().info("Pose hotkey: p records current EEF pose")

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
            self.capture_dataset_photo()
            return True

        if key == "p":
            self.save_pose_to_csv()
            return True

        if key == "v":
            self.toggle_grasp_object()
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

    def ensure_dataset_paths(self):
        if self.dataset_dir is not None and self.zip_path is not None:
            return self.dataset_dir, self.zip_path

        self.dataset_dir, self.zip_path = self.make_unique_dataset_paths()
        self.dataset_dir.mkdir(parents=True, exist_ok=False)
        self.dataset_archived = False

        self.get_logger().info(f"Dataset snapshot folder: {self.dataset_dir}")

        return self.dataset_dir, self.zip_path

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to convert camera image: {exc}")
            return

        with self.dataset_lock:
            self.latest_image = cv_image.copy()
            self.latest_image_stamp = msg.header.stamp

    def capture_dataset_photo(self):
        now = time.monotonic()

        with self.dataset_lock:
            
            if (now - self.last_snapshot_time_s) < 0.1: # --> self.snapshot_debounce_s
                self.get_logger().warn("Snapshot ignored: debounce active")
                return

            self.last_snapshot_time_s = now

            if self.latest_image is None:
                self.get_logger().warn(
                    "Snapshot ignored: no camera frame received yet"
                )
                return

            dataset_dir, _ = self.ensure_dataset_paths()
            image_index = self.saved_count
            cv_image = self.latest_image.copy()

        filename = f"img_{image_index:06d}.jpg"
        filepath = dataset_dir / filename

        try:
            # if image_extension in ["jpg", "jpeg"]:
            #     ok = cv2.imwrite(
            #         str(filepath),
            #         cv_image,
            #         [cv2.IMWRITE_JPEG_QUALITY, 95],
            #     )
            # else:
            #     ok = cv2.imwrite(str(filepath), cv_image)
            # TODO: why if jpg --> 95 quality value ???
            ok = cv2.imwrite(
                str(filepath),
                cv_image,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if not ok:
                self.get_logger().error(f"Failed to write image: {filepath}")
                return

            with self.dataset_lock:
                self.saved_count += 1
                saved_count = self.saved_count

            self.get_logger().info(
                f"Saved photo #{saved_count}: {filepath}"
            )

        except Exception as exc:
            self.get_logger().error(f"Failed to save photo: {exc}")

    def finalize_dataset_archive(self):
        with self.dataset_lock:
            dataset_dir = self.dataset_dir
            zip_path = self.zip_path
            saved_count = self.saved_count
            already_archived = self.dataset_archived

            if saved_count < 1:
                return

            if already_archived:
                return

            self.dataset_archived = True

        self.create_zip_archive(dataset_dir, zip_path, saved_count)

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

    def reload_poses(self):
        self.utils.load_csv_poses(str(self.csv_file))
        self.get_logger().info(f"Reloaded {len(self.utils.poses)} poses from CSV")

    def get_next_counter(self):
        self.reload_poses()
        return len(self.utils.poses) + 1

    @staticmethod
    def ensure_trailing_newline(path: Path):
        if not path.exists() or path.stat().st_size == 0:
            return

        with path.open("rb") as f:
            f.seek(-1, 2)
            last_char = f.read(1)

        if last_char != b"\n":
            with path.open("a", newline="") as f:
                f.write("\n")

    def save_pose_to_csv(self):
        now_s = self.now_s()

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

            self.ensure_trailing_newline(self.csv_file)

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

    def print_main_menu(self):
        print(
            """
Main menu:

  l             -> list parsed CSV poses
  t             -> enter teleop mode
  z             -> capture one dataset photo
  p             -> record current tool0 pose to CSV
  v             -> switch grasp object body/sensor
  <pose_id>     -> switch to joint_trajectory_controller and move to CSV pose
  q / quit      -> exit

During trajectory execution:
  z             -> capture one dataset photo
  p             -> record current tool0 pose to CSV
  v             -> switch grasp object body/sensor
  Ctrl+C        -> exit

Examples:
  1
  100
  224
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
  v      -> switch grasp object body/sensor
  z      -> capture one dataset photo
  p      -> record current tool0 pose to CSV
  t      -> exit teleop and return to main menu
  Ctrl+C -> exit
"""
        )

    def print_available_poses(self):
        self.reload_poses()

        print("\nAvailable poses:")

        def pose_sort_key(item):
            name, pose = item
            try:
                return (0, int(str(pose.get("id", "")).strip()), name)
            except Exception:
                return (1, 0, name)

        for name, pose in sorted(self.utils.poses.items(), key=pose_sort_key):
            pose_id = str(pose.get("id", "")).strip()
            print(
                f"  id={pose_id}  {name} -> "
                f"x={pose['x']:.4f}, y={pose['y']:.4f}, z={pose['z']:.4f}"
            )
        print("")

    def execute_pose(self, pose_name: str, pose: dict) -> bool:
        self.get_logger().info(
            "During execution: press z to capture one image, p to record pose"
        )

        old_settings = self.enable_keyboard_hotkeys()

        try:
            return self.motion.execute_pose(
                pose_name,
                pose,
                wait_fn=lambda future, timeout_s, label: self.utils.wait_future(
                    future,
                    timeout_s,
                    label,
                    tick_fn=self.process_pending_global_hotkeys,
                ),
            )
        finally:
            self.restore_keyboard_hotkeys(old_settings)

    def publish_servo_command(self):
        with self.lock:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame

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
    
    def toggle_grasp_object(self):
        grasp_objects = ["body", "sensor", "mid", "wire5"]

        try:
            current_index = grasp_objects.index(self.selected_grasp_object)
        except ValueError:
            current_index = 0

        self.selected_grasp_object = grasp_objects[
            (current_index + 1) % len(grasp_objects)
        ]

        self.get_logger().info(
            f"Selected grasp object: {self.selected_grasp_object}"
        )

    def toggle_gripper(self):
        if self.gripper_busy:
            self.get_logger().warn("Gripper command already in progress")
            return

        if not self.gripper_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn("Gripper action server is not available")
            return

        selected_grasp_object = self.selected_grasp_object
        pose_type = "OPEN" if self.gripper_is_closed else "CLOSE"
        target_position = self.get_gripper_position(pose_type, selected_grasp_object)

        if target_position is None:
            return

        label = pose_type.lower()

        goal = ParallelGripperCommand.Goal()
        goal.command.position = [target_position]

        self.gripper_busy = True
        self.get_logger().info(
            f"Gripper {label}: object={selected_grasp_object}, "
            f"position={target_position:.4f}"
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

    def enter_teleop_mode(self):
        self.get_logger().info("Aligning tool0 to z_ground before teleop...")

        if not self.utils.switch_to_trajectory_mode():
            self.get_logger().error("Could not switch to trajectory mode for alignment")
            return

        # TODO: check again later
        # if not self.motion.align_tool_to_ground():
        #     self.get_logger().error("Could not align tool0 to z_ground before teleop")
        #     return

        self.get_logger().info("Switching to teleop controller mode...")

        if not self.utils.switch_to_teleop_mode():
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
                    elif key == "v":
                        self.toggle_grasp_object()
                    elif key == "z":
                        self.capture_dataset_photo()
                    elif key == "p":
                        self.publish_stop()
                        self.save_pose_to_csv()
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
                self.capture_dataset_photo()
                continue

            if command == "p":
                self.save_pose_to_csv()
                continue

            if command == "v":
                self.toggle_grasp_object()
                continue

            if command == "t":
                self.enter_teleop_mode()
                self.print_main_menu()
                continue

            self.reload_poses()
            pose = self.utils.get_pose_by_id(command, log_level="warn")
            pose_name = self.utils.get_name_by_id(command)

            if pose is None:
                self.get_logger().warn(
                    f"Unknown command or pose id: {command}. "
                    f"Use 'l' to list poses, 't' for teleop, 'z' for one image, "
                    f"'p' for pose, 'v' for grasp object, or 'q' to quit."
                )
                continue

            self.get_logger().info(
                "Switching to trajectory controller mode before pose execution..."
            )

            if not self.utils.switch_to_trajectory_mode():
                self.get_logger().error("Could not switch to trajectory controller mode")
                continue

            self.execute_pose(pose_name, pose)

    def stop(self):
        self.running = False
        self.in_teleop = False
        self.publish_stop()
        self.finalize_dataset_archive()


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