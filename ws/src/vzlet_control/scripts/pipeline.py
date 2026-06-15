#!/home/sonieth2/vzlet_ur3e/ur_rtde_scripts/venv/bin/python3

import argparse
import time

import rclpy
from control_msgs.action import ParallelGripperCommand
from controller_manager_msgs.srv import SwitchController
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TwistStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ServoCommandType
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from pipeline_motion import MotionController
from pipeline_vision import VisionProcessor

from pipeline_types import (
    ACTION_PICK,
    ACTION_PLACE,
    CIRCLE_DETECTION_YOLO,
    DEFAULT_CSV_FILE,
    DEFAULT_YOLO_MODEL_FILE,
    FINAL_ZONE,
    INITIAL_ZONE,
)

class IntegratedPickPipeline(Node):
    def __init__(self):
        super().__init__("pick_pipe")

        # Frames / robot config.
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("zone_pose_csv", DEFAULT_CSV_FILE)

        # MoveIt config.
        self.declare_parameter("planner_id", "RRTConnect")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("velocity_scaling", 0.1)
        self.declare_parameter("acceleration_scaling", 0.1)
        self.declare_parameter("position_tolerance", 0.005)
        self.declare_parameter("orientation_tolerance", 0.01)

        # Ground plane.
        self.declare_parameter("add_ground_plane", True)
        self.declare_parameter("ground_plane_z", -0.05)
        self.declare_parameter("ground_plane_thickness", 0.04)

        # Visual servo topics / camera.
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("twist_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("debug_image_topic", "/circle_centering/debug_image")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")

        # Controller switching for MoveIt Servo.
        self.declare_parameter(
            "switch_controller_service",
            "/controller_manager/switch_controller",
        )
        self.declare_parameter(
            "trajectory_controller",
            "joint_trajectory_controller",
        )
        self.declare_parameter(
            "servo_controller",
            "forward_position_controller",
        )
        self.declare_parameter(
            "controller_switch_timeout",
            5.0,
        )
        self.declare_parameter(
            "servo_command_type_service",
            "/servo_node/switch_command_type",
        )
        self.declare_parameter(
            "servo_command_type",
            1,
        )

        # D435 color camera intrinsics.
        self.declare_parameter("fx", 609.5712890625)
        self.declare_parameter("fy", 608.7792358398438)
        self.declare_parameter("cx", 318.21075439453125)
        self.declare_parameter("cy", 243.92738342285156)

        # Visual servo behavior.
        self.declare_parameter("target_plane_z_base", 0.03)
        self.declare_parameter("linear_gain", 6.0)
        self.declare_parameter("max_linear_speed", 0.2)
        self.declare_parameter("pixel_deadband", 2.0)
        self.declare_parameter("servo_timeout", 90.0)
        self.declare_parameter("servo_settle_time", 1.0)
        self.declare_parameter("stop_when_lost", True)
        self.declare_parameter("log_period_s", 1.0)

        # Circle detection.
        self.declare_parameter("circle_detection_method", CIRCLE_DETECTION_YOLO)
        self.declare_parameter("min_radius_px", 45.0)
        self.declare_parameter("max_radius_px", 60.0)
        self.declare_parameter("min_circularity", 0.1)
        self.declare_parameter("min_mean_brightness", 50.0)
        self.declare_parameter("yolo_model_path", DEFAULT_YOLO_MODEL_FILE)

        # Grasp.
        self.declare_parameter("z_offset", 0.15 + 0.004) #  length from tool0 + cells offset
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

        self.switch_controller_client = self.create_client(
            SwitchController,
            self.get_parameter("switch_controller_service").value,
        )

        self.servo_command_type_client = self.create_client(
            ServoCommandType,
            self.get_parameter("servo_command_type_service").value,
        )

        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.twist_pub = self.create_publisher(
            TwistStamped,
            self.get_parameter("twist_topic").value,
            10,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            self.get_parameter("debug_image_topic").value,
            10,
        )

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

    # -----------------------------
    # Pipeline orchestration
    # -----------------------------
    def run_pipeline(self) -> bool:
        self.get_logger().info("Waiting for MoveGroup action server...")
        self.movegroup_client.wait_for_server()
        self.get_logger().info("MoveGroup connected")

        self.get_logger().info("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        self.get_logger().info("Gripper connected")

        self.get_logger().info("Waiting for controller switch service...")
        self.switch_controller_client.wait_for_service()
        self.get_logger().info("Controller switch service connected")

        self.get_logger().info("Waiting for Servo command-type service...")
        self.servo_command_type_client.wait_for_service()
        self.get_logger().info("Servo command-type service connected")

        if bool(self.get_parameter("add_ground_plane").value):
            self.publish_ground_plane()

        stages = [
            ("start zone pose", lambda: self.motion.move_to_zone(INITIAL_ZONE)),
            ("tool alignment", self.motion.align_tool_to_ground),
            ("visual servo", lambda: self.vision.run_visual_servo_stage(CIRCLE_DETECTION_YOLO, yolo_target_class="sensor")),
            ("grasp-pick", lambda: self.motion.execute_grasp_sequence(ACTION_PICK)),
            ("final zone pose", lambda: self.motion.move_to_zone(FINAL_ZONE)),
            # ("final zone pose", lambda: self.motion.move_to_zone(FINAL_ZONE, constraint="z_ground")), # TODO fix it
            ("visual servo place", lambda: self.vision.run_visual_servo_stage(CIRCLE_DETECTION_YOLO, yolo_target_class="cell")),
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="One-node integrated UR pick pipeline")
    parser.add_argument("--zone-pose-csv", default=None)
    parser.add_argument("--servo-timeout", type=float, default=None)
    parser.add_argument("--servo-settle-time", type=float, default=None)
    parser.add_argument("--z-offset", type=float, default=None)
    return parser.parse_known_args(argv)


def main(argv=None):
    cli_args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = IntegratedPickPipeline()

    # Convenience CLI overrides. ROS parameters still work normally.
    overrides = []
    if cli_args.zone_pose_csv is not None:
        overrides.append(("zone_pose_csv", cli_args.zone_pose_csv))
    if cli_args.servo_timeout is not None:
        overrides.append(("servo_timeout", cli_args.servo_timeout))
    if cli_args.servo_settle_time is not None:
        overrides.append(("servo_settle_time", cli_args.servo_settle_time))
    if cli_args.z_offset is not None:
        overrides.append(("z_offset", cli_args.z_offset))

    for name, value in overrides:
        node.set_parameters([Parameter(name, value=value)])

    try:
        success = node.run_pipeline()
        return 0 if success else 1
    except KeyboardInterrupt:
        node.get_logger().warn("Pipeline interrupted")
        node.motion.publish_zero_twist(reason="shutdown")
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())