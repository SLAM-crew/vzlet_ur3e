from control_msgs.action import ParallelGripperCommand
from cv_bridge import CvBridge
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import PlanningScene
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class BaseRobotNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)

        self.use_background_executor = False

        self.get_logger().info(f"Initializing {node_name} Base Components...")

        self._declare_robot_parameters()
        self._extract_robot_parameters()
        self._initialize_core_components()

    def _declare_robot_parameters(self):
        # Frames / Group
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("group_name", "ur_manipulator")

        # CSV / camera
        self.declare_parameter("zone_pose_csv", "/home/sonieth2/vzlet_ur3e/ws/zone_poses_floor.csv")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")

        # MoveIt Planning & Pipelines
        self.declare_parameter("move_action", "/move_action")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planner_id", "RRTConnect")
        self.declare_parameter("pilz_pipeline_id", "pilz_industrial_motion_planner")
        self.declare_parameter("pilz_planner_id", "LIN")
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("position_tolerance", 0.001)
        self.declare_parameter("orientation_tolerance", 0.005)

        # Scaling
        self.declare_parameter("velocity_scaling", 0.5)
        self.declare_parameter("acceleration_scaling", 0.1)
        self.declare_parameter("ompl_velocity_scaling", 0.5)
        self.declare_parameter("ompl_acceleration_scaling", 0.1)
        self.declare_parameter("pilz_velocity_scaling", 0.6)
        self.declare_parameter("pilz_acceleration_scaling", 0.15)

        # Controllers // TODO:refactor
        self.declare_parameter("controller_switch_service", "/controller_manager/switch_controller")
        self.declare_parameter("trajectory_controller", "joint_trajectory_controller")
        self.declare_parameter("servo_controller", "forward_position_controller")
        self.declare_parameter("controller_switch_timeout", 5.0)
        self.declare_parameter("servo_command_type_service", "/servo_node/switch_command_type")
        self.declare_parameter("servo_command_type", 1)

        # Gripper
        self.declare_parameter("gripper_action", "/gripper_controller/gripper_cmd")
        self.declare_parameter("gripper_body_close_position", 0.011)
        self.declare_parameter("gripper_sensor_close_position", 0.016)
        self.declare_parameter("gripper_open_position", 0.006)
        self.declare_parameter("gripper_max_effort", 0.0)

        # Grasp // TODO: tune these parameters and remove hardcoding
        self.declare_parameter("z_offset_body", 0.165)
        self.declare_parameter("z_offset_sensor_pick", 0.15)
        self.declare_parameter("z_offset_sensor_place", 0.18)
        self.declare_parameter("z_offset_piezo", 0.1525)

        # YOLO voting parameters.
        self.declare_parameter("yolo_vote_frames", 2)
        self.declare_parameter("yolo_vote_max_center_dist_px", 5.0)
        self.declare_parameter("yolo_vote_min_conf", 0.89)
        self.declare_parameter("yolo_vote_frame_timeout_s", 3.0)
        self.declare_parameter("yolo_vote_debug_dir", "vote_detect")

        # D435 RGB camera intrinsics
        self.declare_parameter("fx", 602.873352)
        self.declare_parameter("fy", 600.606750)
        self.declare_parameter("cx", 304.549032)
        self.declare_parameter("cy", 269.791445)
        self.declare_parameter("yolo_model_path", "/home/sonieth2/vzlet_ur3e/ws/models/vzlet_ver8.pt")

    def _extract_robot_parameters(self):
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.group_name = str(self.get_parameter("group_name").value)

        self.zone_pose_csv = str(self.get_parameter("zone_pose_csv").value)
        self.image_topic = str(self.get_parameter("image_topic").value)

        self.move_action = str(self.get_parameter("move_action").value)
        self.pipeline_id = str(self.get_parameter("pipeline_id").value)
        self.planner_id = str(self.get_parameter("planner_id").value)
        self.pilz_pipeline_id = str(self.get_parameter("pilz_pipeline_id").value)
        self.pilz_planner_id = str(self.get_parameter("pilz_planner_id").value)
        self.planning_attempts = int(self.get_parameter("planning_attempts").value)
        self.allowed_planning_time = float(self.get_parameter("allowed_planning_time").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.orientation_tolerance = float(self.get_parameter("orientation_tolerance").value)

        self.velocity_scaling = float(self.get_parameter("velocity_scaling").value)
        self.acceleration_scaling = float(self.get_parameter("acceleration_scaling").value)
        self.ompl_velocity_scaling = float(self.get_parameter("ompl_velocity_scaling").value)
        self.ompl_acceleration_scaling = float(self.get_parameter("ompl_acceleration_scaling").value)
        self.pilz_velocity_scaling = float(self.get_parameter("pilz_velocity_scaling").value)
        self.pilz_acceleration_scaling = float(self.get_parameter("pilz_acceleration_scaling").value)

        self.controller_switch_service = str(self.get_parameter("controller_switch_service").value)
        self.trajectory_controller = str(self.get_parameter("trajectory_controller").value)
        self.servo_controller = str(self.get_parameter("servo_controller").value)
        self.controller_switch_timeout = float(self.get_parameter("controller_switch_timeout").value)
        self.servo_command_type_service = str(self.get_parameter("servo_command_type_service").value)
        self.servo_command_type = int(self.get_parameter("servo_command_type").value)

        self.gripper_action = str(self.get_parameter("gripper_action").value)
        self.gripper_body_close_position = float(self.get_parameter("gripper_body_close_position").value)
        self.gripper_sensor_close_position = float(self.get_parameter("gripper_sensor_close_position").value)
        self.gripper_open_position = float(self.get_parameter("gripper_open_position").value)
        self.gripper_max_effort = float(self.get_parameter("gripper_max_effort").value)

        self.z_offset_body = float(self.get_parameter("z_offset_body").value)
        self.z_offset_sensor_pick = float(self.get_parameter("z_offset_sensor_pick").value)
        self.z_offset_sensor_place = float(self.get_parameter("z_offset_sensor_place").value)
        self.z_offset_piezo = float(self.get_parameter("z_offset_piezo").value)

        self.yolo_vote_frames = int(self.get_parameter("yolo_vote_frames").value)
        self.yolo_vote_max_center_dist_px = float(self.get_parameter("yolo_vote_max_center_dist_px").value)
        self.yolo_vote_min_conf = float(self.get_parameter("yolo_vote_min_conf").value)
        self.yolo_vote_frame_timeout_s = float(self.get_parameter("yolo_vote_frame_timeout_s").value)
        self.yolo_vote_debug_dir = str(self.get_parameter("yolo_vote_debug_dir").value)

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)
        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)

    def _initialize_core_components(self):
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        self.bridge = CvBridge()

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.movegroup_client = ActionClient(
            self,
            MoveGroup,
            self.move_action,
        )
    
        self.gripper_client = ActionClient(
            self,
            ParallelGripperCommand,
            self.gripper_action,
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9