#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
import threading

from ultralytics import YOLO

from flask import Flask, render_template_string, Response, jsonify, request

#############################################
# Flask 앱 초기화 (단 하나만 선언)
#############################################
app = Flask(__name__)

#############################################
# 전역 변수: ROS2 원본 영상 (ImageSubscriber)
#############################################
cv_image = None
cv_lock = threading.Lock()

#############################################
# 전역 변수: YOLO 추적 영상
#############################################
latest_frame = None
frame_lock = threading.Lock()
latest_detections = []
detections_lock = threading.Lock()

#############################################
# ROS2 노드 1: ImageSubscriber (원본 영상 구독)
#############################################
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/robot/camera/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        global cv_image, cv_lock
        try:
            with cv_lock:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
        cv2.waitKey(1)

#############################################
# ROS2 노드 2: YOLOTrackingNode (YOLO 추적)
#############################################
class YOLOTrackingNode(Node):
    def __init__(self):
        super().__init__('yolo_tracking_node')
        self.publisher_ = self.create_publisher(CompressedImage, 'AMR_image', 10)
        self.selected_coord_publisher = self.create_publisher(Point, 'selected_caller_coord', 10)
        self.bridge = CvBridge()

        # YOLOv8 모델 로드 – 모델 파일 경로를 수정하세요.
        self.model = YOLO('/home/seungrok/standing_best.pt')

        # USB 카메라 열기 – 여기서는 인덱스 2를 사용 (실제 환경에 맞게 변경)
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 필요 시 FOURCC 설정: self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Homography 사용 안 함 (필요하면 추가)
        self.H = None

        # 별도 스레드에서 주기적으로 프레임 읽기 및 YOLO 추적
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Cannot read frame from camera.")
            return

        # YOLOv8 track() 호출: 이 경우, frame 하나를 소스로 전달하여 추적 결과를 얻습니다.
        results = self.model.track(source=frame, stream=True, tracker='bytetrack.yaml')
        result = next(results, None)
        if result is None:
            return

        detection_list = []
        boxes = getattr(result, 'boxes', [])
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else None
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = "Unknown"
            if hasattr(self.model.model, 'names') and cls in self.model.model.names:
                class_name = self.model.model.names[cls]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {track_id} {class_name} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detection_list.append({
                'bbox': [x1, y1, x2, y2],
                'track_id': track_id,
                'class_name': class_name,
                'conf': conf
            })

        global latest_frame, frame_lock, latest_detections, detections_lock
        with frame_lock:
            latest_frame = frame.copy()
        with detections_lock:
            latest_detections = detection_list

        # (선택) ROS2 이미지 메시지 퍼블리시
        _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = encoded_img.tobytes()
        self.publisher_.publish(msg)

    def publish_selected_coord(self, px, py):
        self.get_logger().info(f"Publishing coordinate: ({px},{py})")
        point_msg = Point()
        point_msg.x = float(px)
        point_msg.y = float(py)
        point_msg.z = 0.0
        self.selected_coord_publisher.publish(point_msg)

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()

#############################################
# Flask 웹 인터페이스
#############################################
html_page = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ROS2 + YOLO Tracking Interface</title>
  <style>
    .video-container { display: flex; }
    .video-box { margin: 5px; }
    .btn-container { margin-top: 10px; }
    .btn-container button { margin-right: 10px; padding: 8px 16px; }
  </style>
</head>
<body>
  <h1>ROS2 + YOLO Tracking Interface</h1>
  <div class="video-container">
    <div class="video-box">
      <h3>Raw Video (ROS2 ImageSubscriber)</h3>
      <img id="video-feed1" src="/video_feed1" style="width:320px; height:240px;" alt="Raw Video">
    </div>
    <div class="video-box">
      <h3>YOLO Tracking Video</h3>
      <img id="video-feed2" src="/video_feed2" style="width:320px; height:240px; cursor: crosshair;" alt="YOLO Video" onclick="onImageClick(event)">
    </div>
  </div>
  <div class="btn-container">
    <button onclick="onCall()">Call</button>
    <button onclick="onID1()">ID1</button>
    <button onclick="onID2()">ID2</button>
    <button onclick="onEnd()">END</button>
  </div>
  <script>
    let callMode = false;
    function onCall() {
      callMode = true;
      alert("Call mode activated");
    }
    function onEnd() {
      callMode = false;
      alert("Call mode deactivated");
    }
    function onID1() { alert("ID1 clicked"); }
    function onID2() { alert("ID2 clicked"); }
    function onImageClick(event) {
      if (!callMode) return;
      const rect = event.target.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      console.log("Clicked at", x, y);
      fetch('/click_coord', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: x, y: y })
      })
      .then(res => res.json())
      .then(data => { if(data.message) alert(data.message); });
    }
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_page)

# Flask endpoint: Raw video stream (from ImageSubscriber)
def generate_frames_raw():
    global cv_image, cv_lock
    while True:
        with cv_lock:
            if cv_image is not None:
                ret, buffer = cv2.imencode('.jpg', cv_image)
                if ret:
                    frame_bytes = buffer.tobytes()
                else:
                    frame_bytes = b''
            else:
                frame_bytes = b''
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames_raw(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask endpoint: YOLO tracking video stream
def generate_frames_yolo():
    global latest_frame, frame_lock
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                else:
                    frame_bytes = b''
            else:
                frame_bytes = b''
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames_yolo(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click_coord', methods=['POST'])
def click_coord():
    data = request.get_json()
    if not data:
        return jsonify({"message": "no data"}), 400
    x = data.get('x')
    y = data.get('y')
    if x is None or y is None:
        return jsonify({"message": "invalid coords"}), 400
    global latest_detections, detections_lock, ros_node
    found_id = None
    with detections_lock:
        for det in latest_detections:
            x1, y1, x2, y2 = det['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                found_id = det['track_id']
                break
    ros_node.publish_selected_coord(x, y)
    msg = f"Selected object ID = {found_id}" if found_id is not None else "No bounding box at that coordinate, publishing pixel coordinate only."
    return jsonify({"message": msg})

##############################################
# ROS2 노드 실행 (MultiThreadedExecutor)
##############################################
def ros2_main():
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    rclpy.init()
    image_node = ImageSubscriber()
    yolo_node = YOLOTrackingNode()
    executor = MultiThreadedExecutor()
    executor.add_node(image_node)
    executor.add_node(yolo_node)
    try:
        executor.spin()
    finally:
        image_node.destroy_node()
        yolo_node.destroy_node()
        rclpy.shutdown()

##############################################
# ROS2 ImageSubscriber 노드 (원본 영상)
##############################################
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/robot/camera/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        global cv_image, cv_lock
        try:
            with cv_lock:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
        cv2.waitKey(1)

##############################################
# 메인 실행부
##############################################
def main(args=None):
    # ROS2를 별도 스레드에서 실행
    ros2_thread = threading.Thread(target=ros2_main, daemon=True)
    ros2_thread.start()
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
