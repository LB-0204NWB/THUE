import cv2  # OpenCV để xử lý camera
import tkinter as tk  # Tkinter để tạo giao diện GUI
from PIL import Image, ImageOps, ImageTk  # Pillow để xử lý hình ảnh
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work

# Định nghĩa lớp ứng dụng camera
class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Cố định kích thước cửa sổ
        self.window.geometry("978x640")
        self.window.resizable(False, False)

        # Cài đặt hình nền
        self.background_image = Image.open("../python/backgroud.png").resize((978, 640), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(window, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Hiển thị video
        self.video_label = tk.Label(window)
        self.video_label.place(x=125, y=175, width=350, height=250)

        # Hình ảnh thùng rác
        self.bin_image = Image.open("../python/c4e9637861d7d88981c6.png").resize((200, 200), Image.LANCZOS)
        self.bin_photo = ImageTk.PhotoImage(self.bin_image)
        self.bin_label = tk.Label(window, image=self.bin_photo)
        self.bin_label.place(x=650, y=100)

        # Tải model và labels
        self.model = load_model("../python/converted_keras/keras_model.h5", compile=False)
        self.class_names = open("../python/converted_keras/labels.txt", "r").readlines()
        self.input_size = (224, 224)

        # Nút điều khiển camera
        self.start_button_img = Image.open("../python/BAT.png")
        self.start_button_photo = ImageTk.PhotoImage(self.start_button_img)
        self.btn_start = tk.Button(window, image=self.start_button_photo, borderwidth=0, command=self.start_camera)
        self.btn_start.place(x=100, y=500)

        self.stop_button_img = Image.open("../python/TAT.png")
        self.stop_button_photo = ImageTk.PhotoImage(self.stop_button_img)
        self.btn_stop = tk.Button(window, image=self.stop_button_photo, borderwidth=0, command=self.stop_camera)
        self.btn_stop.place(x=350, y=500)

        # Xử lý sự kiện khi đóng ứng dụng
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.cap = None
        self.is_running = False

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Không thể mở camera.")
                return
            self.is_running = True
            self.update()

    def stop_camera(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')

    def update(self):
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Chuẩn bị hình ảnh để dự đoán
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image = ImageOps.fit(image, self.input_size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                # Dự đoán
                prediction = self.model.predict(data, verbose=0)
                index = np.argmax(prediction)
                class_name = self.class_names[index].strip()
                confidence_score = prediction[0][index]

                # Hiển thị kết quả lên khung hình
                cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Hiển thị video
                frame = cv2.resize(frame, (350, 250))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        if self.is_running:
            self.window.after(10, self.update)

    def on_closing(self):
        self.stop_camera()
        self.window.destroy()

# Tạo cửa sổ Tkinter và chạy ứng dụng
root = tk.Tk()
app = CameraApp(root, "Ứng dụng Camera với Dự đoán")
root.mainloop()
