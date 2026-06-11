#!/home/orangevz/vzlet_ur3e/venv/bin/python3
import os
import sys
import time
import zipfile
import select
import termios
import tty
import threading
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge


class DatasetTeleopRecorder(Node):
    def __init__(self):
        super().__init__("dataset_teleop_recorder")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("servo_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("save_every_n", 45)
        self.declare_parameter("output_dir", "dataset_output")
        self.declare_parameter("frame_id", "tool0")
        self.declare_parameter("linear_speed", 0.1)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("image_extension", "jpg")

        self.image_topic = self.get_parameter("image_topic").value
        self.servo_topic = self.get_parameter("servo_topic").value
        self.save_every_n = int(self.get_parameter("save_every_n").value)
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.image_extension = self.get_parameter("image_extension").value.lower()

        if self.save_every_n < 1:
            self.save_every_n = 1

        if self.image_extension not in ["jpg", "jpeg", "png"]:
            self.get_logger().warn("Unsupported image_extension. Falling back to jpg.")
            self.image_extension = "jpg"

        self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.dataset_dir = self.output_dir / f"ds_{self.timestamp}"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.zip_path = self.output_dir / f"ds_{self.timestamp}.zip"

        self.bridge = CvBridge()
        self.image_count = 0
        self.saved_count = 0

        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = self.frame_id

        self.lock = threading.Lock()
        self.running = True

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

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

        self.get_logger().info(f"Recording every {self.save_every_n}th image")
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Servo topic: {self.servo_topic}")
        self.get_logger().info(f"Frame ID: {self.frame_id}")
        self.get_logger().info(f"Saving dataset to: {self.dataset_dir}")
        self.print_controls()

    def print_controls(self):
        print(
            """
Keyboard controls:

  w / s  -> +X / -X
  a / d  -> +Y / -Y
  r / f  -> +Z / -Z

  space  -> stop
  Ctrl+C -> save zip and exit
"""
        )

    def image_callback(self, msg: Image):
        self.image_count += 1

        if self.image_count % self.save_every_n != 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            filename = f"img_{self.saved_count:06d}.{self.image_extension}"
            filepath = self.dataset_dir / filename

            if self.image_extension in ["jpg", "jpeg"]:
                cv2.imwrite(str(filepath), cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(filepath), cv_image)

            self.saved_count += 1

            if self.saved_count % 25 == 0:
                self.get_logger().info(f"Saved {self.saved_count} images")

        except Exception as exc:
            self.get_logger().error(f"Failed to save image: {exc}")

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
                    elif key == "\x03":
                        raise KeyboardInterrupt
                    else:
                        self.set_velocity()

        except KeyboardInterrupt:
            self.running = False
            rclpy.shutdown()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def create_zip_archive(self):
        self.get_logger().info("Creating zip archive...")

        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.dataset_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.dataset_dir.parent)
                    zipf.write(file_path, arcname)

        self.get_logger().info(f"Zip archive created: {self.zip_path}")
        self.get_logger().info(f"Total saved images: {self.saved_count}")

    def stop(self):
        self.running = False
        self.set_velocity()

        # Publish zero velocity a few times before exit
        for _ in range(5):
            self.publish_servo_command()
            time.sleep(0.02)

        # self.create_zip_archive()


def main(args=None):
    rclpy.init(args=args)

    node = DatasetTeleopRecorder()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.create_zip_archive()

    finally:
        node.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()