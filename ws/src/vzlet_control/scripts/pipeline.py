#!/home/sonieth2/vzlet_ur3e/ur_rtde_scripts/venv/bin/python3

import time
import rclpy
from control_msgs.action import ParallelGripperCommand
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from pipeline_motion import MotionController
from pipeline_vision import VisionProcessor

from pipeline_types import (
    ACTION_PICK,
    ACTION_PLACE,
    ZONE_CSV_FILE,
    DEFAULT_YOLO_MODEL_FILE,
    FINAL_ZONE,
    INITIAL_ZONE,
)

class IntegratedPickPipeline(Node):
    def __init__(self):
        super().__init__("pick_pipe")

        # Frames / robot config.
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("zone_pose_csv", ZONE_CSV_FILE)

        # MoveIt config.
        self.declare_parameter("planner_id", "RRTConnect")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("velocity_scaling", 0.5)
        self.declare_parameter("acceleration_scaling", 0.1)
        self.declare_parameter("position_tolerance", 0.001)
        self.declare_parameter("orientation_tolerance", 0.01)

        # Collision objects.
        self.declare_parameter("add_ground_plane", True)
        self.declare_parameter("ground_plane_z", -0.05)
        self.declare_parameter("ground_plane_thickness", 0.04)

        # Camera topic.
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")

        # YOLO voting parameters.
        self.declare_parameter("yolo_vote_frames", 2)
        self.declare_parameter("yolo_vote_max_center_dist_px", 5.0)
        self.declare_parameter("yolo_vote_min_conf", 0.89)
        self.declare_parameter("yolo_vote_frame_timeout_s", 3.0)
        self.declare_parameter("yolo_vote_debug_dir", "vote_detect")

        # D435 color camera intrinsics.
        self.declare_parameter("fx", 602.873352)
        self.declare_parameter("fy", 600.606750)
        self.declare_parameter("cx", 304.549032)
        self.declare_parameter("cy", 269.791445)

        # Circle detection.
        self.declare_parameter("yolo_model_path", DEFAULT_YOLO_MODEL_FILE)

        # Grasp.
        # TODO: tune these parameters and remove hardcoding
        self.declare_parameter("z_offset", 0.15 + 0.004 + 0.02) #  length from tool0 + cells offset
        self.declare_parameter("gripper_close_position", 0.04)
        self.declare_parameter("gripper_open_position", 0.5)
        self.declare_parameter("gripper_max_effort", 0.0)

        self.base_frame = self.get_parameter("base_frame").value
        self.tool_frame = self.get_parameter("tool_frame").value
        self.group_name = self.get_parameter("group_name").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.movegroup_client = ActionClient(self, MoveGroup, "/move_action")
        self.gripper_client = ActionClient(
            self,
            ParallelGripperCommand,
            "/gripper_controller/gripper_cmd",
        )

        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize Subsystems
        self.motion = MotionController(self)
        self.vision = VisionProcessor(self)

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.vision.image_callback,
            10,
        )
  
    def move_to_voted_sensor_grid_pose(self) -> bool:
        pose_name = self.vision.select_yolo_grid_pose(
            target_class_name="sensor",
        )
        if pose_name is None:
            return False
        return self.motion.move_to_zone(pose_name)

    def run_pipeline(self) -> bool:
        self.get_logger().info("Waiting for MoveGroup action server...")
        self.movegroup_client.wait_for_server()
        self.get_logger().info("MoveGroup connected")

        self.get_logger().info("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        self.get_logger().info("Gripper connected")

        if bool(self.get_parameter("add_ground_plane").value):
            self.publish_ground_plane()

        stages = [
        ("start zone pose", lambda: self.motion.move_to_zone(INITIAL_ZONE)),
        ("tool alignment", self.motion.align_tool_to_ground),
        ("voted sensor grid pose", self.move_to_voted_sensor_grid_pose),
        ("grasp-pick", lambda: self.motion.execute_grasp_sequence(ACTION_PICK)),
        ("final zone pose", lambda: self.motion.move_to_zone(FINAL_ZONE)),
        ("grasp-place", lambda: self.motion.execute_grasp_sequence(ACTION_PLACE)),
    ]

        for name, fn in stages:
            self.get_logger().info(f"=== Starting stage: {name} ===")
            ok = fn()
            if not ok:
                self.get_logger().error(f"Pipeline stopped: stage failed: {name}")
                return False
            self.get_logger().info(f"=== Finished stage: {name} ===")
            time.sleep(0.5)

        self.get_logger().info("Pipeline complete")
        return True

    def publish_ground_plane(self):
        scene = PlanningScene()
        scene.is_diff = True

        collision = CollisionObject()
        collision.header.frame_id = self.base_frame
        collision.header.stamp = self.get_clock().now().to_msg()
        collision.id = "ground_plane"

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        thickness = float(self.get_parameter("ground_plane_thickness").value)
        box.dimensions = [2.0, 2.0, thickness]

        pose = Pose()
        pose.position.z = float(self.get_parameter("ground_plane_z").value)
        pose.orientation.w = 1.0

        collision.primitives.append(box)
        collision.primitive_poses.append(pose)
        collision.operation = CollisionObject.ADD
        scene.world.collision_objects.append(collision)

        self.scene_pub.publish(scene)
        self.get_logger().info("Ground plane added")
        time.sleep(1.0)
    
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

def main():
    rclpy.init()
    node = IntegratedPickPipeline()

    try:
        success = node.run_pipeline()
        return 0 if success else 1
    except KeyboardInterrupt:
        node.get_logger().warn("Pipeline interrupted")
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())