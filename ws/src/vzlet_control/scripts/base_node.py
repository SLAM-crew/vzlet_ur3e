import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from moveit_msgs.action import MoveGroup
from control_msgs.action import ParallelGripperCommand
from tf2_ros import Buffer, TransformListener

class BaseRobotNode(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        
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

        # MoveIt Planning & Pipelines
        self.declare_parameter("move_action", "/move_action")
        self.declare_parameter("planner_id", "RRTConnect")
        self.declare_parameter("pipeline_id", "ompl")
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("position_tolerance", 0.001)
        self.declare_parameter("orientation_tolerance", 0.01)

        # Scaling
        self.declare_parameter("ompl_velocity_scaling", 0.5)
        self.declare_parameter("ompl_acceleration_scaling", 0.1)
        self.declare_parameter("pilz_pipeline_id", "pilz_industrial_motion_planner")
        self.declare_parameter("pilz_planner_id", "LIN")
        self.declare_parameter("pilz_velocity_scaling", 0.6)
        self.declare_parameter("pilz_acceleration_scaling", 0.15)

        # Controllers
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


        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")

    def _extract_robot_parameters(self):
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tool_frame = str(self.get_parameter("tool_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.group_name = str(self.get_parameter("group_name").value)
        
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.gripper_action = str(self.get_parameter("gripper_action").value)

    def _initialize_core_components(self):
        # TF2 setup
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action Clients
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