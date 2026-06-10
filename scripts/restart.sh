#!/bin/bash

source /opt/ros/jazzy/setup.bash

ros2 service call /dashboard_client/close_safety_popup std_srvs/srv/Trigger "{}"

ros2 service call /dashboard_client/restart_safety std_srvs/srv/Trigger "{}"

ros2 service call /dashboard_client/power_on std_srvs/srv/Trigger "{}"

ros2 service call /dashboard_client/brake_release std_srvs/srv/Trigger "{}"

ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger "{}"
