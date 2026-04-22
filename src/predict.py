import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ===== LOAD MODEL =====
MODEL_PATH = r"D:\deeplearning\deeplearning\models\best_resnet_age_model.h5"

if not os.path.exists(MODEL_PATH):
    print(" Không tìm thấy model:", MODEL_PATH)
    exit()

model = load_model(MODEL_PATH)

# ===== LẤY SIZE TỪ MODEL =====
input_shape = model.input_shape  # (None, h, w, 3)
IMG_SIZE = input_shape[1]

print(" Model input shape:", input_shape)

# ===== PREDICT FUNCTION =====
def predict_age(img_path):
    if not os.path.exists(img_path):
        print(" Không tìm thấy ảnh:", img_path)
        return
    
    img = cv2.imread(img_path)

    if img is None:
        print("Không đọc được ảnh")
        return

    # Resize size model
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # preprocessing ResNet
    img = preprocess_input(img)

    # Expand dims
    img = np.expand_dims(img, axis=0)

    print("👉 Input shape:", img.shape)

    # Predict
    pred = model.predict(img, verbose=0)

    # giá trị thô
    print("Raw prediction:", pred[0][0])

    #Scale lại tuổi thật
    age = int(pred[0][0] * 116)

    print(f"Tuổi dự đoán: {age}")


# ===== RUN =====
if __name__ == "__main__":
    predict_age(r"D:\deeplearning\deeplearning\test\105_1_0_20170112213507183.jpg.chip.jpg")