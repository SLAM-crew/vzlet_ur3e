import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ur_robot_driver'), 'launch', 'ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur3e',
            'robot_ip': '192.168.31.59',
            'kinematics_params_file': '/home/sonieth2/vzlet_ur3e/config/ur3e_calibration.yaml',
            'launch_rviz': 'false',
            'initial_joint_controller': 'joint_trajectory_controller',
        }.items()
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ur_moveit_config'), 'launch', 'ur_moveit.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur3e'
        }.items()
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'depth_module.depth_profile': '1280x720x30',
            'pointcloud.enable': 'true'
        }.items()
    )

    # delayed_moveit = TimerAction(
    #     period=5.0,
    #     actions=[moveit_launch]
    # )

    # delayed_realsense = TimerAction(
    #     period=10.0,
    #     actions=[realsense_launch]
    # )

    return LaunchDescription([
        ur_launch,
        realsense_launch
    ])