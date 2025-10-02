import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import time

class CameraDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Khởi tạo mô hình YOLO
        self.model = YOLO(model_path)
        self.cap = None
        self.imgtk = None

    def start_camera(self, device_id=0):
        """Khởi tạo camera từ device_id (mặc định là 0 cho webcam chính)"""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(device_id)
        return self.cap.isOpened()

    def start_video(self, video_path):
        """Khởi tạo video từ đường dẫn tệp"""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        return self.cap.isOpened()

    def detect_frame(self):
        """Phát hiện đối tượng trong khung hình hiện tại"""
        if self.cap is None:
            return None, None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        start = time.time()
        results = self.model(frame)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + 1e-6)

        # Chuyển đổi sang RGB và resize
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((850, 600))
        self.imgtk = ImageTk.PhotoImage(image=img)

        return self.imgtk, fps, len(results[0].boxes)

    def release(self):
        """Giải phóng tài nguyên camera/video"""
        if self.cap:
            self.cap.release()
        self.cap = None
