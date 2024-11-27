import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model  # Keras để load mô hình

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Thiết lập kích thước cửa sổ
        self.window.geometry("978x640")
        self.window.resizable(False, False)

        # Thiết lập hình nền
        self.background_image = Image.open("/home/long/Desktop/DOAN/python/backgroud.png")
        self.background_image = self.background_image.resize((978, 640), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(window, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Khu vực hiển thị video
        self.video_label = tk.Label(window)
        self.video_label.place(x=125, y=175, width=350, height=250)

        # Khu vực hiển thị kết quả dự đoán
        self.result_label = tk.Label(window, text="", font=("Helvetica", 16), bg="white", fg="black")
        self.result_label.place(x=650, y=400, width=300, height=50)

        # Tải mô hình Keras
        self.model = load_model('../python/converted_keras/keras_model.h5')  # Đường dẫn đến file model.h5
        self.input_size = (64, 64)  # Kích thước đầu vào của mô hình (tuỳ vào mô hình)

        # Biến quản lý camera
        self.cap = None
        self.is_running = False

        # Nút điều khiển
        self.start_button_img = Image.open("/home/long/Desktop/DOAN/python/BAT.png")
        self.start_button_photo = ImageTk.PhotoImage(self.start_button_img)

        self.stop_button_img = Image.open("/home/long/Desktop/DOAN/python/TAT.png")
        self.stop_button_photo = ImageTk.PhotoImage(self.stop_button_img)

        self.btn_start = tk.Button(window, image=self.start_button_photo, borderwidth=0, command=self.start_camera)
        self.btn_start.place(x=100, y=500)

        self.btn_stop = tk.Button(window, image=self.stop_button_photo, borderwidth=0, command=self.stop_camera)
        self.btn_stop.place(x=350, y=500)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_camera(self):
        """Bật camera."""
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.update()

    def stop_camera(self):
        """Tắt camera."""
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')
            self.result_label.config(text="")

    def predict_frame(self, frame):
        """Dự đoán trên khung hình."""
        # Resize khung hình theo kích thước đầu vào của mô hình
        resized_frame = cv2.resize(frame, self.input_size)
        normalized_frame = resized_frame / 255.0  # Chuẩn hoá pixel về [0, 1]
        input_data = np.expand_dims(normalized_frame, axis=0)  # Thêm batch dimension

        # Dự đoán với mô hình
        prediction = self.model.predict(input_data)
        predicted_label = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất
        return predicted_label

    def update(self):
        """Cập nhật luồng video và dự đoán."""
        if self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize và chuyển đổi định dạng cho Tkinter
                frame_resized = cv2.resize(frame, (350, 250))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Dự đoán với mô hình
                predicted_label = self.predict_frame(frame)
                self.result_label.config(text=f"Dự đoán: {predicted_label}")  # Hiển thị nhãn dự đoán

        # Lặp lại hàm update mỗi 10ms nếu camera đang chạy
        if self.is_running:
            self.window.after(10, self.update)

    def on_closing(self):
        """Xử lý khi đóng ứng dụng."""
        self.stop_camera()
        self.window.destroy()

# Tạo cửa sổ Tkinter và chạy ứng dụng
root = tk.Tk()
app = CameraApp(root, "Camera with Keras Prediction")
root.mainloop()
