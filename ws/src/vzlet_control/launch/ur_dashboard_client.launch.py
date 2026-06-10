from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
            "dashboard_receive_timeout",
            default_value="20.0",
            description="Timeout that the dashboard client will wait for a response from the robot.",
        )
    )

    # Initialize Arguments
    robot_ip = LaunchConfiguration("robot_ip")
    dashboard_receive_timeout = LaunchConfiguration("dashboard_receive_timeout")

    dashboard_client_node = Node(
        package="ur_robot_driver",
        executable="dashboard_client",
        name="dashboard_client",
        output="screen",
        emulate_tty=True,
        parameters=[
            {"robot_ip": robot_ip},
            {"receive_timeout": dashboard_receive_timeout},
        ],
    )

    return LaunchDescription(declared_arguments + [dashboard_client_node])
