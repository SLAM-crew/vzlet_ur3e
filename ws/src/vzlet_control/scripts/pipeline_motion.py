import copy
import csv
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import (
    PoseStamped,
    Quaternion,

)
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive

from pipeline_types import ACTION_PICK, ACTION_PLACE


class MotionController:
    def __init__(self, node):
        self.node = node
        self.last_log_time = self.node.get_clock().now()

    def get_current_tool0_pose(
        self,
        timeout_s: float = 3.0,
        retry_period_s: float = 0.05,
    ) -> Optional[PoseStamped]:
        base_frame = self.node.get_parameter("base_frame").value
        tool_frame = self.node.get_parameter("tool_frame").value

        start_s = self.now_s()
        last_error = None

        while rclpy.ok() and (self.now_s() - start_s) < timeout_s:
            try:
                tf_base_tool = self.node.tf_buffer.lookup_transform(
                    base_frame,
                    tool_frame,
                    rclpy.time.Time(),
                )

                pose = PoseStamped()
                pose.header.frame_id = base_frame
                pose.header.stamp = self.node.get_clock().now().to_msg()

                pose.pose.position.x = tf_base_tool.transform.translation.x
                pose.pose.position.y = tf_base_tool.transform.translation.y
                pose.pose.position.z = tf_base_tool.transform.translation.z
                pose.pose.orientation = tf_base_tool.transform.rotation

                return pose

            except Exception as exc:
                last_error = exc
                rclpy.spin_once(self.node, timeout_sec=retry_period_s)

        self.node.get_logger().warn(
            f"Could not get current {tool_frame} pose in {base_frame} "
            f"after {timeout_s:.1f}s: {last_error}"
        )
        return None

    def create_move_goal(self, target_pose: PoseStamped, path_constraints: Optional[Constraints] = None) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = self.node.get_parameter("group_name").value
        request.pipeline_id = str(self.node.get_parameter("pipeline_id").value)
        request.planner_id = str(self.node.get_parameter("planner_id").value)
        request.num_planning_attempts = int(self.node.get_parameter("planning_attempts").value)
        request.allowed_planning_time = float(self.node.get_parameter("allowed_planning_time").value)
        request.max_velocity_scaling_factor = float(self.node.get_parameter("velocity_scaling").value)
        request.max_acceleration_scaling_factor = float(self.node.get_parameter("acceleration_scaling").value)

        base_frame = self.node.get_parameter("base_frame").value
        tool_frame = self.node.get_parameter("tool_frame").value

        constraints = Constraints()

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = base_frame
        pos_constraint.link_name = tool_frame
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [float(self.node.get_parameter("position_tolerance").value)]
        volume = BoundingVolume()
        volume.primitives.append(primitive)
        volume.primitive_poses.append(target_pose.pose)
        pos_constraint.constraint_region = volume
        pos_constraint.weight = 1.0

        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = base_frame
        orient_constraint.link_name = tool_frame
        orient_constraint.orientation = target_pose.pose.orientation
        tol = float(self.node.get_parameter("orientation_tolerance").value)
        orient_constraint.absolute_x_axis_tolerance = tol
        orient_constraint.absolute_y_axis_tolerance = tol
        orient_constraint.absolute_z_axis_tolerance = tol
        orient_constraint.weight = 1.0

        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(orient_constraint)
        request.goal_constraints.append(constraints)
        if path_constraints is not None:
            request.path_constraints = path_constraints

        goal.request = request
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def execute_move(self, target_pose: PoseStamped, label: str, path_constraints: Optional[Constraints] = None) -> bool:
        target_pose.header.stamp = self.node.get_clock().now().to_msg()
        self.node.get_logger().info(
            f"{label}: target x={target_pose.pose.position.x:.4f}, y={target_pose.pose.position.y:.4f}, "
            f"z={target_pose.pose.position.z:.4f}, qx={target_pose.pose.orientation.x:.4f}, "
            f"qy={target_pose.pose.orientation.y:.4f}, qz={target_pose.pose.orientation.z:.4f}, "
            f"qw={target_pose.pose.orientation.w:.4f}"
        )
        goal = self.create_move_goal(target_pose, path_constraints=path_constraints)
        send_goal_future = self.node.movegroup_client.send_goal_async(goal)
        
        rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=30.0)
        
        if not send_goal_future.done():
            self.node.get_logger().error(f"{label}: timed out waiting goal acceptance")
            return False

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.node.get_logger().error(f"{label}: goal handle is None")
            return False
        if not goal_handle.accepted:
            self.node.get_logger().error(f"{label}: goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=300.0)
        
        if not result_future.done():
            self.node.get_logger().error(f"{label}: timed out waiting execution result")
            return False

        result = result_future.result()
        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f"{label}: motion executed")
            return True

        status = None if result is None else result.status
        self.node.get_logger().error(f"{label}: execution failed with status {status}")
        return False

    def command_gripper(self, position: float, label: Optional[str] = None) -> bool:
        if label is None:
            open_pos = float(self.node.get_parameter("gripper_open_position").value)
            close_pos = float(self.node.get_parameter("gripper_close_position").value)
            label = "open" if abs(position - open_pos) < abs(position - close_pos) else "close"

        self.node.get_logger().info(f"Gripper {label}: position={position:.4f}")
        goal = ParallelGripperCommand.Goal()
        goal.command.position = [position]

        send_goal_future = self.node.gripper_client.send_goal_async(goal)
        
        rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=30.0)
        
        if not send_goal_future.done():
            self.node.get_logger().error(f"Gripper {label}: timed out waiting goal acceptance")
            return False

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.node.get_logger().error(f"Gripper {label}: goal handle is None")
            return False
        if not goal_handle.accepted:
            self.node.get_logger().error(f"Gripper {label}: goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=30.0)
        
        if not result_future.done():
            self.node.get_logger().error(f"Gripper {label}: timed out waiting result")
            return False

        result = result_future.result()
        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f"Gripper {label}: succeeded")
            return True

        status = None if result is None else result.status
        self.node.get_logger().error(f"Gripper {label}: failed with status {status}")
        return True

    def execute_grasp_sequence(self, action: str) -> bool:
        start_pose = self.get_current_tool0_pose()
        if start_pose is None:
            return False

        z_offset = abs(float(self.node.get_parameter("z_offset").value))
        lowered_pose = copy.deepcopy(start_pose)
        lowered_pose.pose.position.z = z_offset
        if action == ACTION_PICK:
            pre_grasp_pos = float(self.node.get_parameter("gripper_open_position").value)
            post_grasp_pos = float(self.node.get_parameter("gripper_close_position").value)
        elif action == ACTION_PLACE:
            pre_grasp_pos = float(self.node.get_parameter("gripper_close_position").value)
            post_grasp_pos = float(self.node.get_parameter("gripper_open_position").value)

        if action == ACTION_PICK and not self.command_gripper(pre_grasp_pos): return False
        if not self.execute_move(lowered_pose, f"Move down for {action}"): return False
        if not self.command_gripper(post_grasp_pos): return False
        if not self.execute_move(start_pose, "Return to upper pose"): return False

        return True

    def create_tool0_ground_orientation_constraint(self, orientation: Quaternion) -> Constraints:
        constraint = OrientationConstraint()
        constraint.header.frame_id = self.node.get_parameter("base_frame").value
        constraint.link_name = self.node.get_parameter("tool_frame").value
        constraint.orientation = orientation
        tol = float(self.node.get_parameter("orientation_tolerance").value)
        constraint.absolute_x_axis_tolerance = tol
        constraint.absolute_y_axis_tolerance = tol
        constraint.absolute_z_axis_tolerance = tol
        constraint.weight = 1.0

        constraints = Constraints()
        constraints.orientation_constraints.append(constraint)
        return constraints

    def quaternion_to_rotation_matrix(self, q: Quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ])

    def rotation_matrix_to_quaternion(self, matrix):
        m = matrix
        trace = np.trace(m)
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        q = Quaternion()
        q.x = qx
        q.y = qy
        q.z = qz
        q.w = qw
        return q

    def make_tool0_z_face_ground_orientation(self, current_orientation: Quaternion):
        current_rotation = self.quaternion_to_rotation_matrix(current_orientation)
        current_tool_x = current_rotation[:, 0]

        desired_tool_z = np.array([0.0, 0.0, -1.0])
        desired_tool_x = np.array([current_tool_x[0], current_tool_x[1], 0.0])

        if np.linalg.norm(desired_tool_x) < 1e-6:
            desired_tool_x = np.array([1.0, 0.0, 0.0])
        else:
            desired_tool_x = desired_tool_x / np.linalg.norm(desired_tool_x)

        desired_tool_y = np.cross(desired_tool_z, desired_tool_x)
        desired_tool_y = desired_tool_y / np.linalg.norm(desired_tool_y)

        desired_rotation = np.column_stack((
            desired_tool_x,
            desired_tool_y,
            desired_tool_z,
        ))

        return self.rotation_matrix_to_quaternion(desired_rotation)

    def align_tool_to_ground(self) -> bool:
        current_pose = self.get_current_tool0_pose(timeout_s=5.0)

        if not current_pose: return False

        target_pose = copy.deepcopy(current_pose)
        target_pose.pose.orientation = self.make_tool0_z_face_ground_orientation(current_pose.pose.orientation)
        return self.execute_move(target_pose, "Align tool0 to ground")

    def load_zone_poses(self, csv_file_path: str) -> list:
        csv_file = Path(csv_file_path)
        poses = []
        with csv_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {
                    (key or "").strip(): (value.strip() if isinstance(value, str) else value)
                    for key, value in row.items()
                    if key is not None
                }
                poses.append({
                    "name": normalized.get("name"),
                    "id": int(normalized["id"]),
                    "x": float(normalized["x"]),
                    "y": float(normalized["y"]),
                    "z": float(normalized["z"]),
                    "qx": float(normalized["qx"]),
                    "qy": float(normalized["qy"]),
                    "qz": float(normalized["qz"]),
                    "qw": float(normalized["qw"]),
                })
        return poses
    
    def move_to_zone(self,  pose_name: str, constraint: Optional[str] = None) -> bool:
        try:
            poses = self.load_zone_poses(self.node.get_parameter("zone_pose_csv").value)
        except Exception as exc:
            self.node.get_logger().error(f"Could not load zone poses: {exc}")
            return False

        if not poses:
            self.node.get_logger().error("No zone poses found")
            return False

        selected_pose = next((p for p in poses if str(p.get("name", "")).strip() == pose_name), None)

        if selected_pose is None:
            available_names = ", ".join(str(p.get("name", "")).strip() for p in poses if p.get("name"))
            self.node.get_logger().error(
                f"Zone pose '{pose_name}' was not found in CSV; available names: {available_names}"
            )
            return False

        path_constraints = None
        label_prefix = "zone pose"

        if constraint == "z_ground":
            current_pose = self.get_current_tool0_pose()
            if current_pose is None:
                self.node.get_logger().error("Could not retrieve current pose for orientation constraint")
                return False

            constrained_orientation = self.make_tool0_z_face_ground_orientation(
                current_pose.pose.orientation
            )
            path_constraints = self.create_tool0_ground_orientation_constraint(constrained_orientation)
            label_prefix = "constrained zone pose"

        target = PoseStamped()
        target.header.frame_id = self.node.get_parameter("base_frame").value
        target.header.stamp = self.node.get_clock().now().to_msg()
        target.pose.position.x = selected_pose["x"]
        target.pose.position.y = selected_pose["y"]
        target.pose.position.z = selected_pose["z"]
        target.pose.orientation.x = selected_pose["qx"]
        target.pose.orientation.y = selected_pose["qy"]
        target.pose.orientation.z = selected_pose["qz"]
        target.pose.orientation.w = selected_pose["qw"]

        label = f"{label_prefix} {selected_pose['name']} (#{selected_pose['id']})"
        
        if not self.execute_move(target, label, path_constraints=path_constraints):
            return False
        
        time.sleep(1.5)
        return True
    
    def now_s(self):
        return self.node.get_clock().now().nanoseconds * 1e-9
