from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir


def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            description="IP address by which the robot can be reached.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_mock_hardware",
            default_value="false",
            description="Start robot with mock hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "mock_sensor_commands",
            default_value="false",
            description="Enable mock command interfaces for sensors used for simple simulations. "
            "Used only if 'use_mock_hardware' parameter is true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Initially loaded robot controller.",
            choices=[
                "scaled_joint_trajectory_controller",
                "joint_trajectory_controller",
                "forward_velocity_controller",
                "forward_position_controller",
                "freedrive_mode_controller",
                "passthrough_trajectory_controller",
            ],
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "activate_joint_controller",
            default_value="true",
            description="Activate loaded joint controller.",
        )
    )

    # Initialize Arguments
    robot_ip = LaunchConfiguration("robot_ip")
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    mock_sensor_commands = LaunchConfiguration("mock_sensor_commands")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")

    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), "/ur_control.launch.py"]),
        launch_arguments={
            "ur_type": "ur3e",
            "robot_ip": robot_ip,
            "use_mock_hardware": use_mock_hardware,
            "mock_sensor_commands": mock_sensor_commands,
            "initial_joint_controller": initial_joint_controller,
            "activate_joint_controller": activate_joint_controller,
        }.items(),
    )

    return LaunchDescription(declared_arguments + [base_launch])