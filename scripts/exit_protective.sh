ros2 service call /dashboard_client/unlock_protective_stop std_srvs/srv/Trigger {}
ros2 service call /dashboard_client/close_safety_popup std_srvs/srv/Trigger {}
ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger {}