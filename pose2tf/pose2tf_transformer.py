import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import tf_transformations as tf
import numpy as np


class Pose2Tf(Node):
    def __init__(self):
        super().__init__('relative_tf_publisher')

        # Parameters
        self.declare_parameter('pose_camera_topic', '/vrpn_mocap/d435/pose')
        self.declare_parameter('pose_person_topic', '/vrpn_mocap/luka/pose')
        self.declare_parameter('camera_frame_id', 'd435')
        self.declare_parameter('person_frame_id', 'person')

        pose_camera_topic = self.get_parameter('pose_camera_topic').get_parameter_value().string_value
        pose_person_topic = self.get_parameter('pose_person_topic').get_parameter_value().string_value
        self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value
        self.person_frame_id = self.get_parameter('person_frame_id').get_parameter_value().string_value

        # Storage for latest poses
        self.pose_camera = None
        self.pose_person = None

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.sub_cam = self.create_subscription(PoseStamped, pose_camera_topic, self.cam_callback, qos_profile)
        self.sub_person = self.create_subscription(PoseStamped, pose_person_topic, self.person_callback, qos_profile)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Distance publisher
        self.distance_pub = self.create_publisher(Float64, 'd435_to_person_distance', 10)

        self.get_logger().info(f"Subscribed to {pose_camera_topic} and {pose_person_topic}")
        self.get_logger().info(f"Publishing TF: {self.camera_frame_id} -> {self.person_frame_id}")
        self.get_logger().info(f"Publishing distance on topic: d435_to_person_distance")

    def cam_callback(self, msg: PoseStamped):
        self.pose_camera = msg
        self.publish_relative()

    def person_callback(self, msg: PoseStamped):
        self.pose_person = msg
        self.publish_relative()

    def publish_relative(self):
        if self.pose_camera is None or self.pose_person is None:
            return

        # Compute relative: d435 -> person
        T_wc = self.pose_to_matrix(self.pose_camera.pose)
        T_wp = self.pose_to_matrix(self.pose_person.pose)

        T_cw = np.linalg.inv(T_wc)       # camera->world
        T_cp = np.dot(T_cw, T_wp)        # camera->person

        # Convert back to translation + quaternion
        trans = T_cp[0:3, 3]
        quat = tf.quaternion_from_matrix(T_cp)

        # Publish TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame_id
        t.child_frame_id = self.person_frame_id
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

        # Publish distance
        distance = np.linalg.norm(trans)
        msg = Float64()
        msg.data = distance*100
        self.distance_pub.publish(msg)

    def pose_to_matrix(self, pose):
        trans = [pose.position.x, pose.position.y, pose.position.z]
        rot = [pose.orientation.x, pose.orientation.y,
               pose.orientation.z, pose.orientation.w]
        return np.dot(tf.translation_matrix(trans), tf.quaternion_matrix(rot))


def main(args=None):
    rclpy.init(args=args)
    node = Pose2Tf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
