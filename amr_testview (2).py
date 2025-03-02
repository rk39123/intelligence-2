#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class TrackedImageViewer(Node):
    def __init__(self):
        super().__init__('tracked_image_viewer')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/tracked_image',  # Turtlebot에서 퍼블리시하는 토픽 이름
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # 압축 이미지를 OpenCV 이미지로 변환
        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        cv2.imshow("Tracked Image (from Turtlebot)", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TrackedImageViewer()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()
