import os
import sys
import argparse
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

# Đường dẫn tuyệt đối dựa trên vị trí file predict.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_age_model.h5")

IMG_SIZE = 224

def predict_age(image_path):
    """Dự đoán tuổi dựa trên thuật toán tìm khuôn mặt."""
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh '{image_path}'")
        return None

    model = load_model(MODEL_PATH)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc file ảnh '{image_path}'")
        return None

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    min_size = (int(img.shape[1] * 0.15), int(img.shape[0] * 0.15))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)

    if len(faces) == 0:
        face_cascade_def = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade_def.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)

    if len(faces) == 0:
        print("Cảnh báo: Không tìm thấy khuôn mặt! Sử dụng toàn bộ ảnh.")
        faces = [(0, 0, img.shape[1], img.shape[0])]
    else:
        print(f"Phát hiện {len(faces)} khuôn mặt trong ảnh.")
        # Lọc lấy khuôn mặt lớn nhất để tránh nhận nhầm các vật thể tĩnh ở background
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]

    first_age = None

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        
        # Tiền xử lý riêng phần khuôn mặt
        img_array = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)
        age = int(pred[0][0] * 100)
        if first_age is None:
            first_age = age

        print(f"- Khuôn mặt ({w}x{h}) dự đoán: {age} tuổi")

        font_scale = max(0.5, w / 200.0)
        thickness = max(1, int(font_scale * 2))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness * 2)
        cv2.putText(img, f"Age: {age}", (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    display_img = img.copy()
    img_h, img_w = display_img.shape[:2]
    max_h, max_w = 800, 1200
    
    if img_h > max_h or img_w > max_w:
        scaling = min(max_h / float(img_h), max_w / float(img_w))
        display_img = cv2.resize(display_img, (int(img_w * scaling), int(img_h * scaling)), interpolation=cv2.INTER_AREA)

    cv2.imshow("Age Prediction", display_img)
    print("Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return first_age


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán tuổi từ ảnh khuôn mặt")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn tới ảnh cần dự đoán")
    args = parser.parse_args()

    image_path = args.image

    if image_path is None:
        # Mở hộp thoại (File Dialog) để người dùng chọn ảnh thoải mái
        root = tk.Tk()
        root.withdraw()  # Ẩn cửa sổ command của Tkinter
        print("Đang mở hộp thoại chọn ảnh...")
        image_path = filedialog.askopenfilename(
            title="Chọn file ảnh khuôn mặt",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not image_path:
            print("Đã hủy chọn ảnh. Kết thúc chương trình.")
            sys.exit(0)

    predict_age(image_path)