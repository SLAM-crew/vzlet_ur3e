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
from pipeline_utils import PipelineUtils

class IntegratedPickPipeline(Node):
    def __init__(self):
        super().__init__("pick_pipe")

        # Frames / robot config.
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("zone_pose_csv", "/home/sonieth2/vzlet_ur3e/ws/zone_poses_floor.csv")

        # MoveIt config.
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planner_id", "RRTConnect")
        self.declare_parameter("pilz_pipeline_id", "pilz_industrial_motion_planner")
        self.declare_parameter("pilz_planner_id", "LIN")
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)

        self.declare_parameter("pilz_velocity_scaling", 0.6)
        self.declare_parameter("pilz_acceleration_scaling", 0.15)
        self.declare_parameter("ompl_velocity_scaling", 0.5)
        self.declare_parameter("ompl_acceleration_scaling", 0.1)

        self.declare_parameter("position_tolerance", 0.001)
        self.declare_parameter("orientation_tolerance", 0.005)

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
        self.declare_parameter("yolo_model_path", "/home/sonieth2/vzlet_ur3e/ws/models/vzlet_ver8.pt")

        # Grasp.
        # TODO: tune these parameters and remove hardcoding
        self.declare_parameter("z_offset_body", 0.165)
        self.declare_parameter("z_offset_sensor_pick", 0.15)
        self.declare_parameter("z_offset_sensor_place", 0.18)
        self.declare_parameter("z_offset_piezo", 0.1525)


        self.declare_parameter("gripper_body_close_position", 0.011)
        self.declare_parameter("gripper_sensor_close_position", 0.016)
        self.declare_parameter("gripper_open_position", 0.006)
        self.declare_parameter("gripper_max_effort", 0.0)

        ### TODO refactor
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
        ###

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
        self.utils = PipelineUtils(self)

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.vision.image_callback,
            10,
        )
  
    def move_to_voted_grid_pose(self, class_type: str) -> bool:
        pose_name = self.vision.select_yolo_grid_pose(
            target_class_name=class_type,
        )
        if pose_name is None:
            return False
        return self.motion.move_to_zone(pose_name, "z_ground")

    def run_pipeline(self) -> bool:
        self.get_logger().info("Waiting for MoveGroup action server...")
        self.movegroup_client.wait_for_server()
        self.get_logger().info("MoveGroup connected")

        # self.get_logger().info("Waiting for gripper action server...")
        # self.gripper_client.wait_for_server()
        # self.get_logger().info("Gripper connected")

        if bool(self.get_parameter("add_ground_plane").value):
            self.publish_ground_plane()

        self.get_logger().info("Switching to trajectory controller mode")
        if not self.utils.switch_to_trajectory_mode():
            self.get_logger().error("Could not switch to trajectory controller mode")
            return False
        
        try:
            self.utils.load_csv_poses()
        except Exception as exc:
            self.get_logger().error(f"Could not load zone poses: {exc}")
            return False

        stages = [
        # ("pick zone: body", lambda: self.motion.move_to_zone("BODY_PICK_ZONE", constraint="z_ground")),
        # ("tool alignment", lambda: self.motion.align_tool_to_ground()),
        # ("voted grid pose: body", lambda: self.move_to_voted_grid_pose("body")),
        # ("grasp-pick: body", lambda: self.motion.execute_grasp_sequence("ACTION_PICK", "body")), 
        # ("body cell zone", lambda: self.motion.move_to_zone("BODY_CELL_ZONE")),
        # ("grasp-place: body", lambda: self.motion.execute_grasp_sequence("ACTION_PLACE", "body")),
        ("sensor pick zone", lambda: self.motion.move_to_zone("SENSOR_PICK_ZONE", constraint="z_ground")),
        ("voted grid pose: piezo", lambda: self.move_to_voted_grid_pose("piezo")),
        ("grasp-pick: piezo", lambda: self.motion.execute_pneumatic_grasp_sequence("ACTION_PICK")),
        ("grasp-place: piezo", lambda: self.motion.execute_pneumatic_grasp_sequence("ACTION_PLACE")),

        # ("tool alignment", lambda: self.motion.align_tool_to_ground()),
        # ("voted grid pose: sensor", lambda: self.move_to_voted_grid_pose("sensor")),
        # ("grasp-pick: sensor", lambda: self.motion.execute_grasp_sequence("ACTION_PICK", "sensor")),
        # ("body cell zone", lambda: self.motion.move_to_zone("BODY_CELL_ZONE", constraint="z_ground")),
        # ("grasp-place: sensor", lambda: self.motion.execute_grasp_sequence("ACTION_PLACE", "sensor")),

    ]

        for name, fn in stages:
            self.get_logger().info(f"=== Starting stage: {name} ===")
            ok = fn()
            if not ok:
                self.get_logger().error(f"Stage failed: {name}")
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