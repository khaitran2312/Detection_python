import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2, os, time
from camera import CameraHandler
from config import Config

# Load YOLO model
model = YOLO(Config.MODEL_PATH)

# Khởi tạo window
root = tb.Window(themename="cosmo")
root.title("🚀 Object Detection App")
root.geometry(f"{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}")

# ========== ẢNH NỀN ==========
bg_image = Image.open("Background.png").resize((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tb.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # full background

# ========== MENU DỌC ==========
menu_frame = tb.Frame(root, bootstyle="secondary", width=Config.MENU_WIDTH)
menu_frame.pack(side="left", fill="y")

# ========== KHU VỰC HIỂN THỊ KẾT QUẢ ==========
lbl = tb.Label(root, background="", borderwidth=0)
lbl.place(x=Config.MENU_WIDTH, y=0, relwidth=0.8, relheight=0.95)

# Set ảnh mặc định để tránh nền trắng
default_img = Image.open("Background.png").resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
default_photo = ImageTk.PhotoImage(default_img)
lbl.config(image=default_photo)
lbl.image = default_photo

# Status bar
status = tb.Label(root, text="Sẵn sàng", anchor="w", bootstyle="dark")
status.pack(side="bottom", fill="x")

# Initialize camera handler
camera_handler = CameraHandler()

def update_status(msg):
    status.config(text=msg)

# ========== XỬ LÝ ẢNH ==========
def detect_image():
    global camera_handler
    camera_handler.stop_camera()
    file_path = filedialog.askopenfilename(filetypes=[("Ảnh", Config.IMAGE_EXTENSIONS)])
    if not file_path: return
    results = model(file_path, save=True, project=Config.OUTPUT_DIR, name=Config.RESULTS_SUBDIR, exist_ok=True)
    output_file = os.path.join(Config.OUTPUT_DIR, Config.RESULTS_SUBDIR, os.path.basename(file_path))
    img = Image.open(output_file).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
    imgtk = ImageTk.PhotoImage(img)
    lbl.config(image=imgtk)
    lbl.image = imgtk
    update_status(f"Ảnh: {os.path.basename(file_path)} | Đối tượng: {len(results[0].boxes)}")

# ========== XỬ LÝ VIDEO ==========
def detect_video():
    global camera_handler
    camera_handler.stop_camera()
    file_path = filedialog.askopenfilename(filetypes=[("Video", Config.VIDEO_EXTENSIONS)])
    if not file_path: return
    cap = cv2.VideoCapture(file_path)
    update_status(f"Video: {os.path.basename(file_path)}")
    show_frame()

# ========== XỬ LÝ CAMERA ==========
def detect_camera():
    global camera_handler
    if camera_handler.start_camera():
        update_status("Camera bật")
        show_frame()

def show_frame():
    global camera_handler
    frame = camera_handler.get_frame()
    if frame is not None:
        start = time.time()
        results = model(frame)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + Config.MIN_FPS)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)
        update_status(f"Camera/Video | FPS: {fps:.2f} | Đối tượng: {len(results[0].boxes)}")
    lbl.after(Config.FRAME_DELAY, show_frame)

# ========== NÚT TRONG MENU ==========
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

# ========== CHẠY APP ==========
try:
    root.mainloop()
finally:
    camera_handler.release()
