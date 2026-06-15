#!/home/sonieth2/vzlet_ur3e/ur_rtde_scripts/venv/bin/python3
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DatasetTeleopRecorder(Node):
    def __init__(self):
        super().__init__("dataset_teleop_recorder")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("save_every_n", 45)
        self.declare_parameter("output_dir", "dataset_output")
        self.declare_parameter("image_extension", "jpg")

        self.image_topic = self.get_parameter("image_topic").value
        self.save_every_n = int(self.get_parameter("save_every_n").value)
        self.output_dir = Path(self.get_parameter("output_dir").value)
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

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(f"Recording every {self.save_every_n}th image")
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Saving dataset to: {self.dataset_dir}")

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
        pass


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