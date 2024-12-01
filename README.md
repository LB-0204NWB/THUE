```python

"""
Hệ Thống Phân Loại Trái Cây Tự Động
------------------------------------
Chương trình sử dụng camera để nhận diện và phân loại trái cây, sau đó gửi tín hiệu điều khiển đến Arduino
để thực hiện phân loại vật lý.

Luồng hoạt động chính:
1. Khởi tạo camera và kết nối Arduino
2. Chụp ảnh liên tục từ camera
3. Xử lý ảnh và dự đoán bằng mô hình AI
4. Đếm số lần nhận diện liên tiếp để đảm bảo độ chính xác
5. Gửi lệnh điều khiển đến Arduino khi đủ điều kiện
6. Hiển thị kết quả lên giao diện

Yêu cầu:
- OpenCV cho xử lý ảnh
- PIL cho hiển thị ảnh
- Keras cho mô hình AI
- PySerial cho giao tiếp Arduino
- Tkinter cho giao diện đồ họa
"""

import cv2
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageOps, ImageTk
import numpy as np
from keras.models import load_model
import serial
import time
from datetime import datetime

class CameraApp:
    def __init__(self, window, window_title):
        """
        Khởi tạo ứng dụng với các thành phần chính:
        - Giao diện người dùng (GUI)
        - Kết nối Arduino
        - Camera
        - Mô hình AI
        - Bộ đếm và theo dõi trạng thái
        """
        self.window = window
        self.window.title(window_title)
        self.window.geometry("978x800")
        self.window.resizable(False, False)

        # Thiết lập ngưỡng nhận diện và bộ đếm
        self.detection_threshold = 20  # Số lần nhận diện liên tiếp cần thiết
        self.detection_counter = 0     # Bộ đếm hiện tại
        self.last_sent_class = None    # Lưu lớp cuối cùng đã gửi đến Arduino

        # Khởi tạo kết nối Arduino
        self._initialize_arduino()

        # Thiết lập giao diện
        self._setup_gui()

        # Khởi tạo mô hình và danh sách lớp
        self._initialize_model()

        # Khởi tạo camera
        self.cap = None
        self.is_running = False
        self.previous_class = None

        # Bắt đầu theo dõi cổng Serial
        self.monitor_serial()

    def _initialize_arduino(self):
        """Thiết lập kết nối với Arduino qua cổng Serial"""
        try:
            self.arduino = serial.Serial('COM3', 9600, timeout=1)
            time.sleep(2)  # Chờ Arduino khởi động
            print("Serial Connection Status:")
            print(f"Port: {self.arduino.port}")
            print(f"Baudrate: {self.arduino.baudrate}")
            print(f"Connected: {self.arduino.is_open}")
            print("Arduino initialized successfully")
        except Exception as e:
            print(f"Arduino Connection Error: {e}")
            self.arduino = None

    def _setup_gui(self):
        """Thiết lập giao diện người dùng với các thành phần:
        - Hình nền
        - Khung video
        - Bảng thống kê
        - Giao diện chat
        - Nút điều khiển
        """
        # Thiết lập hình nền
        self.background_image = Image.open("../python/backgroud.png").resize((978, 800), Image.LANCZOS)
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(self.window, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Khung hiển thị video
        self.video_label = tk.Label(self.window)
        self.video_label.place(x=125, y=175, width=350, height=250)

        # Tạo giao diện chat
        self.create_chat_interface()

        # Thiết lập bảng thống kê
        self._setup_treeview()

        # Thiết lập nút điều khiển camera
        self._setup_camera_controls()

    def _initialize_model(self):
        """Tải mô hình AI và thiết lập các thông số liên quan:
        - Tải mô hình từ file
        - Đọc danh sách tên lớp
        - Tạo mapping tên hiển thị
        - Khởi tạo bộ đếm
        """
        # Tải mô hình và labels
        self.model = load_model("../python/converted_keras/keras_model.h5", compile=False)
        self.class_names = [class_name.strip() for class_name in open("../python/converted_keras/labels.txt", "r").readlines()]
        
        # Mapping tên hiển thị cho các lớp
        self.display_names = {
            self.class_names[0]: "Khong",  # Không có vật thể
            self.class_names[1]: "Cam",    # Quả cam
            self.class_names[2]: "Chanh",  # Quả chanh
            self.class_names[3]: "Nho",    # Quả nho
            self.class_names[4]: "Dâu",    # Quả dâu
            self.class_names[5]: "Lê",     # Quả lê
            self.class_names[6]: "Xoài",   # Quả xoài
            self.class_names[7]: "Bưởi",   # Quả bưởi
            self.class_names[8]: "Quýt",   # Quả quýt
        }

        # In thông tin các lớp đã tải
        print("\nLoaded Classes:")
        for i, name in enumerate(self.class_names):
            display_name = self.display_names.get(name, name)
            print(f"Index {i}: {name} -> {display_name}")
            
        self.input_size = (224, 224)  # Kích thước đầu vào chuẩn cho mô hình
        
        # Khởi tạo bộ đếm cho từng lớp
        self.class_counts = {class_name: 0 for class_name in self.class_names}
        self.update_treeview()

    def _setup_treeview(self):
        """Thiết lập bảng thống kê số lượng từng loại trái cây"""
        # Định dạng cho bảng
        self.style = ttk.Style()
        self.style.configure("Treeview", 
                            font=("Arial", 12),
                            rowheight=30,
                            background="#f9f9f9",
                            fieldbackground="#f9f9f9",
                            borderwidth=1)
        self.style.configure("Treeview.Heading", 
                           font=("Arial", 14, "bold"), 
                           foreground="blue")

        # Tạo khung cho bảng
        self.tree_frame = tk.Frame(self.window)
        self.tree_frame.place(x=600, y=200, width=300, height=300)

        # Thêm thanh cuộn
        self.tree_scroll = tk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Tạo bảng
        self.tree = ttk.Treeview(
            self.tree_frame, 
            columns=("Class", "Count"), 
            show="headings", 
            yscrollcommand=self.tree_scroll.set
        )
        self.tree.heading("Class", text="Loại Trái Cây")
        self.tree.heading("Count", text="Số Lượng")
        self.tree.column("Class", anchor="center", width=150)
        self.tree.column("Count", anchor="center", width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree_scroll.config(command=self.tree.yview)

    def create_chat_interface(self):
        """Tạo giao diện chat để hiển thị và gửi nhận dữ liệu với Arduino"""
        # Khung chat chính
        chat_frame = tk.Frame(self.window, bg='white')
        chat_frame.place(x=125, y=600, width=700, height=150)

        # Vùng hiển thị tin nhắn
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, width=60, height=6)
        self.chat_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Khung nhập liệu
        input_frame = tk.Frame(chat_frame, bg='white')
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Ô nhập tin nhắn
        self.message_entry = tk.Entry(input_frame, width=50)
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind('<Return>', lambda e: self.send_message())

        # Nút gửi
        send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        send_button.pack(side=tk.RIGHT)

    def _setup_camera_controls(self):
        """Thiết lập các nút điều khiển camera"""
        # Nút bật camera
        self.start_button_img = Image.open("../python/BAT.png")
        self.start_button_photo = ImageTk.PhotoImage(self.start_button_img)
        self.btn_start = tk.Button(
            self.window, 
            image=self.start_button_photo, 
            borderwidth=0, 
            command=self.start_camera
        )
        self.btn_start.place(x=100, y=500)

        # Nút tắt camera
        self.stop_button_img = Image.open("../python/TAT.png")
        self.stop_button_photo = ImageTk.PhotoImage(self.stop_button_img)
        self.btn_stop = tk.Button(
            self.window, 
            image=self.stop_button_photo, 
            borderwidth=0, 
            command=self.stop_camera
        )
        self.btn_stop.place(x=350, y=500)

    def send_message(self):
        """Gửi tin nhắn thủ công đến Arduino"""
        message = self.message_entry.get().strip()
        if message and self.arduino:
            try:
                command = f"{message}\n"
                self.arduino.write(command.encode())
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.chat_display.insert(tk.END, f"[{timestamp}] Sent: {message}\n")
                self.chat_display.see(tk.END)
                
                self.message_entry.delete(0, tk.END)
            except Exception as e:
                self.chat_display.insert(tk.END, f"Error sending message: {str(e)}\n")
                self.chat_display.see(tk.END)

    def monitor_serial(self):
        """Theo dõi và hiển thị dữ liệu từ Arduino"""
        if self.arduino and self.arduino.in_waiting:
            try:
                message = self.arduino.readline().decode().strip()
                if message:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.chat_display.insert(tk.END, f"[{timestamp}] Received: {message}\n")
                    self.chat_display.see(tk.END)
            except Exception as e:
                print(f"Error reading from serial: {e}")
        
        self.window.after(100, self.monitor_serial)  # Kiểm tra mỗi 100ms

    def send_to_arduino(self, class_name):
        """Gửi lệnh điều khiển đến Arduino dựa trên lớp đã nhận diện"""
        if self.arduino:
            try:
                # Lấy index của lớp và gửi
                class_index = self.class_names.index(class_name)
                command = f"{class_index}\n"
                self.arduino.write(command.encode())
                
                # Hiển thị thông tin gửi
                timestamp = datetime.now().strftime("%H:%M:%S")
                display_name = self.display_names.get(class_name, class_name)
                self.chat_display.insert(tk.END, 
                    f"[{timestamp}] Sent Command: {command.strip()} for {display_name}\n")
                self.chat_display.see(tk.END)
                
                # Đọc phản hồi từ Arduino
                time.sleep(0.1)
                if self.arduino.in_waiting:
                    response = self.arduino.readline().decode().strip()
                    self.chat_display.insert(tk.END, 
                        f"[{timestamp}] Arduino Response: {response}\n")
                    self.chat_display.see(tk.END)
                
            except Exception as e:
                self.chat_display.insert(tk.END, f"Serial Communication Error: {str(e)}\n")
                self.chat_display.see(tk.END)

    def start_camera(self):
        """Khởi động camera và bắt đầu quá trình nhận diện"""
        if not self.is_running:
            self.cap = cv2.VideoCapture(1)  # Camera USB (1) hoặc built-in (0)
            if not self.cap.isOpened():
                print("Không thể mở camera.")
                return
            print("\nCamera Started")
            self.is_running = True
            self.update()  # Bắt đầu vòng lặp cập nhật

    def stop_camera(self):
        """Dừng camera và đóng kết nối Arduino"""
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')
            print("\nCamera Stopped")
        if self.arduino:
            self.arduino.close()
            print("Arduino Connection Closed")
    def update(self):
    """
    Cập nhật liên tục cho quá trình nhận diện:
    1. Chụp và xử lý ảnh từ camera
    2. Thực hiện dự đoán với mô hình
    3. Xử lý kết quả và gửi lệnh nếu thỏa điều kiện
    4. Cập nhật giao diện
    """
    if self.is_running and self.cap.isOpened():
        ret, frame = self.cap.read()
        if ret:
            # Xử lý ảnh đầu vào
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = ImageOps.fit(image, self.input_size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Thực hiện dự đoán
            prediction = self.model.predict(data, verbose=0)
            index = np.argmax(prediction)  # Lấy index của lớp có xác suất cao nhất
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]  # Độ tin cậy của dự đoán

            # Cập nhật bộ đếm nhận diện liên tiếp
            if class_name == self.previous_class:
                self.detection_counter += 1
            else:
                self.detection_counter = 1

            # Xử lý khi đạt ngưỡng nhận diện và là lớp mới
            if self.detection_counter == self.detection_threshold and class_name != self.last_sent_class:
                self.last_sent_class = class_name  # Cập nhật lớp đã gửi
                self.class_counts[class_name] += 1  # Tăng số lượng
                self.update_treeview()  # Cập nhật bảng thống kê
                
                # In thông tin nhận diện
                display_name = self.display_names.get(class_name, class_name)
                print(f"\nNew Classification:")
                print(f"Detected Class: {display_name}")
                print(f"Confidence Score: {confidence_score:.2f}")
                
                # Gửi lệnh điều khiển đến Arduino
                self.send_to_arduino(class_name)

            # Cập nhật lớp hiện tại
            if class_name != self.previous_class:
                self.previous_class = class_name

            # Hiển thị thông tin lên video
            display_name = self.display_names.get(self.previous_class, self.previous_class) if self.previous_class else "Unknown"
            cv2.putText(frame, f"Class: {display_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {self.detection_counter}/{self.detection_threshold}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Cập nhật hiển thị video
            frame = cv2.resize(frame, (350, 250))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Lập lịch cập nhật tiếp theo
        if self.is_running:
            self.window.after(10, self.update)  # Cập nhật mỗi 10ms

    def update_treeview(self):
        """
        Cập nhật bảng thống kê số lượng từng loại trái cây
        """
        self.tree.delete(*self.tree.get_children())  # Xóa dữ liệu cũ
        for class_name, count in self.class_counts.items():
            # Lấy tên hiển thị và cập nhật vào bảng
            display_name = self.display_names.get(class_name, class_name)
            self.tree.insert("", "end", values=(display_name, count))

    def on_closing(self):
        """
        Xử lý khi đóng ứng dụng:
        - Dừng camera
        - Đóng kết nối Arduino
        - Đóng cửa sổ ứng dụng
        """
        self.stop_camera()
        self.window.destroy()

# Khởi tạo và chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "Hệ Thống Phân Loại Trái Cây Tự Động")
    # Thiết lập xử lý khi đóng cửa sổ
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    # Chạy vòng lặp chính
    root.mainloop()
```
