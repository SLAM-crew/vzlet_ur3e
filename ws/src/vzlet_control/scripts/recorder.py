#!/usr/bin/env python3

import csv
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener


CSV_FILE = "recorded_poses_floor.csv"
CSV_HEADERS = ["name", "id", "x", "y", "z", "qx", "qy", "qz", "qw"]


def _normalize_row(row):
    return {
        (key or "").strip(): (value.strip() if isinstance(value, str) else value)
        for key, value in row.items()
        if key is not None
    }


def _read_rows(path):
    if not path.exists():
        return [], []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [_normalize_row(row) for row in reader]
    return fieldnames, rows


def _rewrite_csv_with_name_column(path):
    fieldnames, rows = _read_rows(path)
    normalized_fieldnames = [name.strip() for name in fieldnames]
    if normalized_fieldnames == CSV_HEADERS:
        return

    rewritten_rows = []
    for row in rows:
        pose_id = int(row.get("id", 0))
        rewritten_rows.append({
            "name": row.get("name"),
            "id": pose_id,
            "x": row.get("x"),
            "y": row.get("y"),
            "z": row.get("z"),
            "qx": row.get("qx"),
            "qy": row.get("qy"),
            "qz": row.get("qz"),
            "qw": row.get("qw"),
        })

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rewritten_rows)


def ensure_csv_exists():
    """Create CSV file with headers if it doesn't exist"""
    path = Path(CSV_FILE)
    if path.exists():
        _rewrite_csv_with_name_column(path)
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()


def get_next_counter():
    """Get the next ID number based on existing CSV entries"""
    path = Path(CSV_FILE)
    if not path.exists():
        return 1

    _, rows = _read_rows(path)
    if not rows:
        return 1

    try:
        return max(int(row.get("id", 0)) for row in rows) + 1
    except Exception:
        return 1


class PoseRecorder(Node):

    def __init__(self):
        super().__init__("pose_recorder")

        self.declare_parameter("servo_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("frame_id", "tool0")
        self.declare_parameter("linear_speed", 0.1)
        self.declare_parameter("publish_rate", 30.0)

        self.servo_topic = self.get_parameter("servo_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_recorded = False
        self.running = True

        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id

        self.lock = threading.Lock()

        self.servo_pub = self.create_publisher(
            TwistStamped,
            self.servo_topic,
            10,
        )

        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_servo_command,
        )

        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True,
        )
        self.keyboard_thread.start()

        self.get_logger().info(f"Servo topic: {self.servo_topic}")
        self.get_logger().info(f"Frame ID: {self.frame_id}")
        self.print_controls()

    def print_controls(self):
        print(
            """
Keyboard controls:

  w / s  -> +X / -X
  a / d  -> +Y / -Y
  r / f  -> +Z / -Z

  space  -> record current pose
  Ctrl+C -> exit
"""
        )

    def publish_servo_command(self):
        with self.lock:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            msg.twist.linear.x = self.current_twist.twist.linear.x
            msg.twist.linear.y = self.current_twist.twist.linear.y
            msg.twist.linear.z = self.current_twist.twist.linear.z

        self.servo_pub.publish(msg)

    def set_velocity(self, x=0.0, y=0.0, z=0.0):
        with self.lock:
            self.current_twist.twist.linear.x = x
            self.current_twist.twist.linear.y = y
            self.current_twist.twist.linear.z = z

    def record_current_pose(self):
        if self.pose_recorded:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "tool0",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            counter = get_next_counter()

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

            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(row)

            self.get_logger().info(f"Recorded pose {row['name']} (#{counter}): {row}")
            self.pose_recorded = True

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed, pose not recorded: {e}")

    def keyboard_loop(self):
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)

                    speed = self.linear_speed

                    if key == "w":
                        self.set_velocity(x=speed)
                    elif key == "s":
                        self.set_velocity(x=-speed)
                    elif key == "a":
                        self.set_velocity(y=speed)
                    elif key == "d":
                        self.set_velocity(y=-speed)
                    elif key == "r":
                        self.set_velocity(z=speed)
                    elif key == "f":
                        self.set_velocity(z=-speed)
                    elif key == " ":
                        self.set_velocity()
                        self.record_current_pose()
                    elif key == "\x03":
                        raise KeyboardInterrupt
                    else:
                        self.set_velocity()

        except KeyboardInterrupt:
            self.running = False
            rclpy.shutdown()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def stop(self):
        self.running = False
        self.set_velocity()

        for _ in range(5):
            self.publish_servo_command()
            time.sleep(0.02)


def main():
    rclpy.init()
    ensure_csv_exists()

    node = PoseRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down pose recorder...")
    finally:
        node.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()