#!/home/sonieth2/vzlet_ur3e/ur_rtde_scripts/venv/bin/python3

import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from sensor_msgs.msg import Image
from shape_msgs.msg import SolidPrimitive

from base_node import BaseRobotNode
from pipeline_motion import MotionController
from pipeline_utils import PipelineUtils
from pipeline_vision import VisionProcessor


class IntegratedPickPipeline(BaseRobotNode):
    def __init__(self):
        super().__init__("pick_pipe")

        self.declare_parameter("add_ground_plane", True)
        self.declare_parameter("ground_plane_z", -0.05)
        self.declare_parameter("ground_plane_thickness", 0.04)

        self.add_ground_plane = bool(self.get_parameter("add_ground_plane").value)
        self.ground_plane_z = float(self.get_parameter("ground_plane_z").value)
        self.ground_plane_thickness = float(
            self.get_parameter("ground_plane_thickness").value
        )

        self.utils = PipelineUtils(self)
        self.motion = MotionController(self)
        self.vision = VisionProcessor(self)

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.vision.image_callback,
            10,
        )

    def move_to_voted_grid_pose(self, class_type: str) -> bool:
        pose_name = self.vision.select_yolo_grid_pose(
            target_class_name=class_type,
        )

        if pose_name is None:
            return False

        return self.motion.move_to_zone(pose_name, constraint="z_ground")

    def run_pipeline(self) -> bool:
        self.get_logger().info("Waiting for MoveGroup action server...")
        self.movegroup_client.wait_for_server()
        self.get_logger().info("MoveGroup connected")

        # self.get_logger().info("Waiting for gripper action server...")
        # self.gripper_client.wait_for_server()
        # self.get_logger().info("Gripper connected")

        if self.add_ground_plane:
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
        box.dimensions = [2.0, 2.0, self.ground_plane_thickness]

        pose = Pose()
        pose.position.z = self.ground_plane_z
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