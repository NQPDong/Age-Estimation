import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Định nghĩa nhóm tuổi (đồng bộ với dataset_loader.py)
AGE_GROUPS = {
    0: "0-2 Em bé",
    1: "3-12 Trẻ em",
    2: "13-18 Vị thành niên",
    3: "19-35 Thanh niên",
    4: "36-55 Trung niên",
    5: "55+ Người cao tuổi",
}

# ===== LOAD MODEL =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet_age_model.h5")

if not os.path.exists(MODEL_PATH):
    print(" Không tìm thấy model:", MODEL_PATH)
    exit()

model = load_model(MODEL_PATH)

# ===== LẤY SIZE TỪ MODEL =====
input_shape = model.input_shape  # (None, h, w, 3)
IMG_SIZE = input_shape[1]

print(" Model input shape:", input_shape)

# ===== FACE DETECTOR =====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ===== PREDICT FUNCTION =====
def predict_age(img_path):
    if not os.path.exists(img_path):
        print(" Không tìm thấy ảnh:", img_path)
        return
    
    img = cv2.imread(img_path)

    if img is None:
        print("Không đọc được ảnh")
        return

    # Phát hiện khuôn mặt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        print("⚠️ Không phát hiện khuôn mặt! Dùng toàn bộ ảnh gốc để predict.")
        face_img = img
    else:
        # Lấy khuôn mặt lớn nhất
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Mở rộng vùng cắt thêm 20% để lấy thêm context (trán, cằm)
        pad = int(0.2 * max(w, h))
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        face_img = img[y1:y2, x1:x2]
        print(f"✅ Phát hiện {len(faces)} khuôn mặt. Dùng khuôn mặt lớn nhất ({w}x{h} px).")

    # Lưu ảnh debug khuôn mặt đã cắt để người dùng kiểm tra trực quan
    DEBUG_DIR = os.path.join(BASE_DIR, "test")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_crop_path = os.path.join(DEBUG_DIR, "anh_duoc_cat.png")
    cv2.imwrite(debug_crop_path, face_img)
    print(f"📸 Đã lưu ảnh khuôn mặt cắt được vào: {debug_crop_path} (Hãy mở file này để kiểm tra xem có cắt đúng khuôn mặt không)")

    # Pipeline đồng bộ với training: cvtColor → Downsample (giảm độ sắc nét để khớp với UTKFace) → Resize → preprocess
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Nếu ảnh khuôn mặt cắt được có độ phân giải lớn, hạ độ phân giải xuống 100x100 trước
    # Điều này giúp giảm nhiễu tần số cao (high-frequency) và khớp với độ mịn của tập UTKFace (ảnh gốc chỉ 200x200)
    h_c, w_c = face_img.shape[:2]
    if max(h_c, w_c) > 150:
        face_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_LINEAR)
        
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    face_img = preprocess_input(face_img.astype(np.float32))
    face_img = np.expand_dims(face_img, axis=0)

    # Predict
    pred = model.predict(face_img, verbose=0)

    # Lấy nhóm tuổi có xác suất cao nhất
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx] * 100

    group_name = AGE_GROUPS[class_idx]
    print(f"Nhóm tuổi dự đoán: {group_name}")
    print(f"Độ tin cậy: {confidence:.1f}%")
    print(f"Xác suất các nhóm:")
    for i, prob in enumerate(pred[0]):
        print(f"  {AGE_GROUPS[i]}: {prob*100:.1f}%")


if __name__ == "__main__":
    predict_age(r"D:\Deeplearning\deeplearning\test\5tuoi.jpg")