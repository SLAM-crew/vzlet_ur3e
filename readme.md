

### Teacher Pendant setup 

Complete [Prepare robot and network connection](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/robot_setup.html#prepare-robot-and-network-connection): Robot setup and Network setup.

[Realtime kernel](https://ubuntu.com/blog/enable-real-time-ubuntu) вместо того чтобы билдить по инструкциии, мы нашли что у ubuntu про есть ууже сбилженный что ли
Ubuntu 22 24+  вроде версии только
```
sudo add-apt-repository universe
sudo apt update
sudo apt install ubuntu-realtime
```


realsense ros2 driver: https://github.com/realsenseai/realsense-ros

How to launch?

First - ros2driver
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3e robot_ip:=192.168.31.59 kinematics_params_file:=config/ur3e_calibration.yaml
```

Second - Teacher Pendant --> press `play` button

Third - do `ros2 control switch_controllers --activate scaled_joint_trajectory_controller`

TODO: add third to custom launch file


ID's system for `zone_poses_floor.csv`
200 --> `2` id of the storage ; `00` id of the grid cell place

sudo apt-get install ros-jazzy-trac-ik-kinematics-plugin

Send pose command for joints
```
ros2 action send_goal /scaled_joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
"{
  trajectory: {
    joint_names: [
      shoulder_pan_joint,
      shoulder_lift_joint,
      elbow_joint,
      wrist_1_joint,
      wrist_2_joint,
      wrist_3_joint
    ],
    points: [
      {
        positions: [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        velocities: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        time_from_start: {sec: 5, nanosec: 0}
      }
    ]
  }
}"
```


In case of emergency-btn was pressed:
```
...
[robot_state_helper-3] [INFO] [1779480718.263959502] [robot_state_helper]: The robot is currently in safety mode ROBOT_EMERGENCY_STOP.
[robot_state_helper-3] [INFO] [1779480718.273956357] [robot_state_helper]: The robot is currently in mode POWER_ON.
[robot_state_helper-3] [INFO] [1779480718.305953326] [robot_state_helper]: The robot is currently in safety mode FAULT.
[ros2_control_node-1] [INFO] [1779480718.306645437] [UR_Client_Library:]: Connection to reverse interface dropped.
[ros2_control_node-1] [INFO] [1779480718.306688387] [UR_Client_Library:]: Trajectory disconnect
[ros2_control_node-1] [INFO] [1779480718.309119870] [controller_manager]: Deactivating controllers: [ scaled_joint_trajectory_controller ]
[ros2_control_node-1] [INFO] [1779480718.309264769] [controller_manager]: Requested controller switch from non-realtime loop
[ros2_control_node-1] [INFO] [1779480718.309350447] [controller_manager]: Successfully switched controllers!
...
```

TODO: can turn ON/OFF using ros2 API ? etc -> change robot_state

TODO: create better calibration `ur3e_robot_calibration.yaml` --> they recommend took it in `motion`

### Local setup - ubuntu24
Sources 
[1]() 
[2](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/real_time.html#real-time-setup)
[3](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/installation.html)


Real-time scheduling for linux setup (is it enough?):
```
sudo apt install linux-lowlatency
```

Related binaries:
```
sudo apt install ros-kilted-moveit
sudo apt install ros-kilted-ros2controlcli
sudo apt-get install ros-${ROS_DISTRO}-ur
```

```
git clone -b kilted https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git
```


Изменили safety password на teacher pedant: стал `vz`

Параллельный хват
LECP6P1-LEHZ25K2-14 - power supply dc24 japan
LEHZ25K2-14-S116P1
xact gripper model (LEHZ25K2-14), the PNP controller (LECP6P1)
[Operation manual for LEH Electric Gripper](https://www.smcworld.com/upfiles/manual/en-jp/files/LEHZ-OM00217.pdf)

https://makerbotics.com/product/mb-elc-servo-controller-urt1/

---

About `Universal_Robots_ROS2_Driver/Universal_Robots_ROS2_Driver.kilted.repos` file --> do not copy it somehow:
```
repositories:realsense ros2 driver
  Universal_Robots_ROS2_Description:
    type: git
    url: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
    version: rolling

can be swap with
```
git clone -b rolling https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
```