#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from lifecycle_msgs.srv import GetState
from std_msgs.msg import String
import threading
import sys
import select
import termios
import tty

class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('initial_pose_publisher')
        self.publisher = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.odom_ready = False
        self.tf_ready = False
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)

        self.create_timer(1.0, self.check_robot_ready)

    def tf_callback(self, msg):
        self.tf_ready = True
        # self.get_logger().info('CHECK_ROBOT_READY - /tf : TRUE')

    def odom_callback(self, msg):
        self.odom_ready = True
        # self.get_logger().info('CHECK_ROBOT_READY - /odom : TRUE')

    def check_robot_ready(self):
        self.get_logger().info(f'{self.tf_ready} | {self.odom_ready}')
        if self.tf_ready and self.odom_ready:
            self.get_logger().info('Robot is ready. You can now publish the initial pose.')
            return True
        else:
            self.get_logger().info('Robot is not ready yet. Waiting...')
            return False
    #  -------------------------------------------------------------------------------------

    def publish_initial_pose(self):
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.frame_id = 'map'                            
        initial_pose.header.stamp = self.get_clock().now().to_msg()

        initial_pose.pose.pose.position.x = 0.018750306218862534     
        initial_pose.pose.pose.position.y = 0.09062495827674866     
        initial_pose.pose.pose.position.z = 0.0      

        initial_pose.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=-1.8423227484934082e-06, 
            w=0.9999999999983029 
        )

        initial_pose.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467
]

        self.get_logger().info('ing??')
        self.publisher.publish(initial_pose)
        self.get_logger().info('The initial position is published.')

        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()

    while not node.check_robot_ready():
        rclpy.spin_once(node)
    
    node.get_logger().info('The robot is ready. Publishing the initial position.')
    node.publish_initial_pose()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
