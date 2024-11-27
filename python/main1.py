import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageOps, ImageTk
import numpy as np
from keras.models import load_model

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
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

        # Bảng hiển thị class và số lần
        self.style = ttk.Style()
        self.style.configure("Treeview", 
                             font=("Arial", 12),  # Font chữ chính
                             rowheight=30,       # Chiều cao mỗi dòng
                             background="#f9f9f9",  # Màu nền dòng
                             fieldbackground="#f9f9f9",  # Màu nền tổng thể
                             borderwidth=1)
        self.style.configure("Treeview.Heading", font=("Arial", 14, "bold"), foreground="blue")

        # Tạo bảng Treeview
        self.tree_frame = tk.Frame(window)
        self.tree_frame.place(x=650, y=250, width=300, height=250)

        self.tree_scroll = tk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree = ttk.Treeview(self.tree_frame, columns=("Class", "Count"), show="headings", yscrollcommand=self.tree_scroll.set)
        self.tree.heading("Class", text="Class")
        self.tree.heading("Count", text="Count")
        self.tree.column("Class", anchor="center", width=150)
        self.tree.column("Count", anchor="center", width=100)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree_scroll.config(command=self.tree.yview)

        # Tải model và labels
        self.model = load_model("../python/converted_keras/keras_model.h5", compile=False)
        self.class_names = [class_name.strip() for class_name in open("../python/converted_keras/labels.txt", "r").readlines()]
        self.input_size = (224, 224)

        # Khởi tạo bộ đếm và cập nhật bảng
        self.class_counts = {class_name: 0 for class_name in self.class_names}
        self.update_treeview()

        # Nút điều khiển camera
        self.start_button_img = Image.open("../python/BAT.png")
        self.start_button_photo = ImageTk.PhotoImage(self.start_button_img)
        self.btn_start = tk.Button(window, image=self.start_button_photo, borderwidth=0, command=self.start_camera)
        self.btn_start.place(x=100, y=500)

        self.stop_button_img = Image.open("../python/TAT.png")
        self.stop_button_photo = ImageTk.PhotoImage(self.stop_button_img)
        self.btn_stop = tk.Button(window, image=self.stop_button_photo, borderwidth=0, command=self.stop_camera)
        self.btn_stop.place(x=350, y=500)

        self.cap = None
        self.is_running = False
        self.previous_class = None
        self.frame_classes = []

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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image = ImageOps.fit(image, self.input_size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array

                prediction = self.model.predict(data, verbose=0)
                index = np.argmax(prediction)
                class_name = self.class_names[index]
                confidence_score = prediction[0][index]

                self.frame_classes.append(class_name)
                if len(self.frame_classes) > 3:
                    self.frame_classes.pop(0)

                if len(self.frame_classes) == 3 and all(cls == self.frame_classes[0] for cls in self.frame_classes):
                    if class_name != self.previous_class:
                        self.previous_class = class_name
                        self.class_counts[class_name] += 1
                        self.update_treeview()

                cv2.putText(frame, f"Class: {self.previous_class or 'Unknown'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frame = cv2.resize(frame, (350, 250))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        if self.is_running:
            self.window.after(10, self.update)

    def update_treeview(self):
        self.tree.delete(*self.tree.get_children())
        for class_name, count in self.class_counts.items():
            self.tree.insert("", "end", values=(class_name, count))

    def on_closing(self):
        self.stop_camera()
        self.window.destroy()

# Tạo cửa sổ Tkinter và chạy ứng dụng
root = tk.Tk()
app = CameraApp(root, "Ứng dụng Camera với Bảng Đẹp")
root.mainloop()
