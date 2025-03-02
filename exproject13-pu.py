#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import time

def get_camera_index(max_cam_index=10):
    """
    사용 가능한 카메라 인덱스를 검색합니다.
    0을 제외한 인덱스가 있으면 첫 번째를 반환하고,
    없으면 사용 가능한 첫 번째 인덱스를, 그래도 없으면 0을 반환합니다.
    """
    available_cams = []
    for index in range(max_cam_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cams.append(index)
            cap.release()
    for cam in available_cams:
        if cam != 0:
            return cam
    if available_cams:
        return available_cams[0]
    return 0

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        # CompressedImage 메시지 발행자 생성
        self.publisher = self.create_publisher(CompressedImage, '/camera/image/compressed', 10)
        self.bridge = CvBridge()

        # 사용 가능한 카메라 인덱스 선택
        cam_index = get_camera_index()
        self.get_logger().info(f"Using camera index: {cam_index}")
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            rclpy.shutdown()

        # 10Hz 주기로 프레임을 발행하는 타이머 생성
        self.timer = self.create_timer(0.1, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame")
            return

        # CvBridge를 이용해 OpenCV 이미지(frame)를 CompressedImage 메시지로 변환
        msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
        self.publisher.publish(msg)
        self.get_logger().debug("Published camera frame")

        # 로컬 디버깅을 위해 창에 표시 (필요시)
        cv2.imshow("Camera Publisher", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
