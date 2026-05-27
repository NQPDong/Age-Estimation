import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Định nghĩa nhóm tuổi (đồng bộ với dataset_loader.py)
AGE_GROUPS = {
    0: "0-2 Em be",
    1: "3-12 Tre em",
    2: "13-18 Vi thanh nien",
    3: "19-35 Thanh nien",
    4: "36-55 Trung nien",
    5: "55+ Nguoi cao tuoi",
}

# Đường dẫn tuyệt đối dựa trên vị trí file cam.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet_age_model.h5")

IMG_SIZE = 224  # Đồng bộ với kích thước model đã huấn luyện

# Khởi tạo bộ phát hiện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Đang tải model...")
model = load_model(MODEL_PATH)
print("Đã tải model thành công!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Lỗi: Không thể mở camera!")
    sys.exit(1)

print("Nhấn ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame từ camera!")
        break

    # Phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Mở rộng vùng cắt thêm 20% để lấy thêm context (trán, cằm)
            pad = int(0.2 * max(w, h))
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)
            face_img = frame[y1:y2, x1:x2]

            # Pipeline đồng bộ với predict.py: cvtColor → Downsample → Resize → preprocess
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            h_c, w_c = face_rgb.shape[:2]
            if max(h_c, w_c) > 150:
                face_rgb = cv2.resize(face_rgb, (100, 100), interpolation=cv2.INTER_LINEAR)
            face_rgb = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            face_rgb = preprocess_input(face_rgb.astype(np.float32))
            face_input = np.expand_dims(face_rgb, axis=0)

            pred = model.predict(face_input, verbose=0)
            class_idx = np.argmax(pred[0])
            confidence = pred[0][class_idx] * 100
            group_name = AGE_GROUPS[class_idx]

            # Vẽ khung chữ nhật quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Hiển thị nhãn nhóm tuổi phía trên khung mặt
            label = f"{group_name} ({confidence:.0f}%)"
            label_y = y - 10 if y - 10 > 10 else y + h + 25
            cv2.putText(frame, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # Không phát hiện khuôn mặt — hiển thị thông báo
        cv2.putText(frame, "Khong phat hien khuon mat",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.imshow("Age Classification", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()