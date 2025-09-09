#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import LookupException, ExtrapolationException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
if not hasattr(np, 'float'):
    np.float = float

import tf_transformations as tf


class Pose2Tf(Node):
    def __init__(self):
        super().__init__('pose2tf_minimal')

        # -------- Parameters --------
        self.declare_parameter('pose_marker_topic', '/vrpn_mocap/d435/pose')
        self.declare_parameter('pose_person_topic', '/vrpn_mocap/luka/pose')
        self.declare_parameter('world_frame_id', 'world')
        self.declare_parameter('marker_frame_id', 'marker')
        self.declare_parameter('person_frame_id', 'person')
        self.declare_parameter('camera_frame_id', 'd435')

        # marker->d435 offset in mm (converted to meters)
        self.declare_parameter('offset_x_mm', 50.53)
        self.declare_parameter('offset_y_mm', 46.22)
        self.declare_parameter('offset_z_mm', 57.44)

        world_frame = self.get_parameter('world_frame_id').value
        self.marker_frame = self.get_parameter('marker_frame_id').value
        self.person_frame = self.get_parameter('person_frame_id').value
        self.camera_frame = self.get_parameter('camera_frame_id').value

        # Offset mm -> meters
        ox = self.get_parameter('offset_x_mm').value / 1000.0
        oy = self.get_parameter('offset_y_mm').value / 1000.0
        oz = self.get_parameter('offset_z_mm').value / 1000.0
        self.T_marker_to_d435 = tf.translation_matrix([ox, oy, oz])

        # Storage for latest poses
        self.pose_marker = None
        self.pose_person = None

        # QoS for mocap
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.create_subscription(PoseStamped, self.get_parameter('pose_marker_topic').value, self.marker_cb, qos)
        self.create_subscription(PoseStamped, self.get_parameter('pose_person_topic').value, self.person_cb, qos)

        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # TF Listener for distance calculation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Distance publisher
        self.dist_pub = self.create_publisher(Float64, 'd435_to_person_distance', 10)

        self.get_logger().info(f"Publishing TFs: {world_frame}->{self.marker_frame}, {world_frame}->{self.person_frame}, {self.marker_frame}->{self.camera_frame}")

        # Timer to compute distance periodically
        self.create_timer(0.02, self.publish_distance)  # 50 Hz

    # -------- Callbacks --------
    def marker_cb(self, msg: PoseStamped):
        self.pose_marker = msg
        self.publish_matrix_as_tf(self.pose_to_matrix(msg.pose), 'world', self.marker_frame)

        # marker -> d435 (constant)
        self.publish_matrix_as_tf(self.T_marker_to_d435, self.marker_frame, self.camera_frame)

    def person_cb(self, msg: PoseStamped):
        self.pose_person = msg
        self.publish_matrix_as_tf(self.pose_to_matrix(msg.pose), 'world', self.person_frame)

    # -------- Distance computation --------
    def publish_distance(self):
        try:
            t = self.tf_buffer.lookup_transform(self.camera_frame, self.person_frame, rclpy.time.Time())
            x, y, z = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z
            msg = Float64()
            msg.data = float(np.linalg.norm([x, y, z]))  # meters
            self.dist_pub.publish(msg)
        except (LookupException, ExtrapolationException):
            pass  # ignore if transform not available yet

    # -------- Helpers --------
    def pose_to_matrix(self, pose):
        trans = [pose.position.x, pose.position.y, pose.position.z]
        rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return np.dot(tf.translation_matrix(trans), tf.quaternion_matrix(rot))

    def publish_matrix_as_tf(self, T, parent, child):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(T[0, 3])
        t.transform.translation.y = float(T[1, 3])
        t.transform.translation.z = float(T[2, 3])
        q = tf.quaternion_from_matrix(T)
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(t)


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
