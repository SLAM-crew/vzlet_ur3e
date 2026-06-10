#!/usr/bin/env python3

import csv
from pathlib import Path

import rclpy
from rclpy.node import Node

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Flag to ensure we only record once
        self.pose_recorded = False
        
        # Create timer to lookup TF every 0.1 seconds
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        """Timer callback that looks up TF every 0.1 seconds"""
        
        if self.pose_recorded:
            # Already recorded, just return (timer continues but does nothing)
            return
        
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "tool0",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)  # Short timeout for continuous lookup
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
            self.timer.cancel()

        except Exception as e:
            # Silently fail or log only occasionally to avoid spam
            self.get_logger().debug(f"TF lookup failed: {e}")


def main():
    rclpy.init()
    ensure_csv_exists()
    
    node = PoseRecorder()
    
    # self.get_logger().info("Pose recorder started - waiting for transform...")
    
    # Wait for Ctrl+C
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down pose recorder...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()