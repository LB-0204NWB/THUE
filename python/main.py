from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2  # Thư viện xử lý video

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("../python/converted_keras/keras_model.h5", compile=False)

# Load the labels
class_names = open("../python/converted_keras/labels.txt", "r").readlines()

# Thiết lập camera
cap = cv2.VideoCapture(0)  # Sử dụng camera mặc định (ID = 0)

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

# Thiết lập kích thước đầu vào của mô hình
input_size = (224, 224)

# Vòng lặp xử lý luồng video
while True:
    # Đọc từng khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận khung hình. Thoát...")
        break

    # Chuyển đổi khung hình sang định dạng RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize và chuẩn hóa khung hình
    image = Image.fromarray(frame_rgb)
    image = ImageOps.fit(image, input_size, Image.Resampling.LANCZOS)  # Resize và crop từ giữa
    image_array = np.asarray(image)  # Chuyển sang mảng numpy
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1  # Chuẩn hóa [-1, 1]

    # Chuẩn bị đầu vào cho mô hình
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Dự đoán với mô hình
    prediction = model.predict(data,verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Hiển thị dự đoán trên khung hình
    cv2.putText(frame, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị khung hình với dự đoán
    cv2.imshow("Camera", frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
