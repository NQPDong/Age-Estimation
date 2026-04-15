import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Đường dẫn tuyệt đối dựa trên vị trí file cam.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_age_model.h5")

IMG_SIZE = 224  # Đồng bộ với kích thước model đã huấn luyện

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

    # Tiền xử lý ảnh — đồng bộ với pipeline huấn luyện
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    age = model.predict(img, verbose=0)
    predicted_age = int(age[0][0] * 100)  # Giải chuẩn hóa (nhân 100)

    cv2.putText(frame, f"Age: {predicted_age}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Age Prediction", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()