import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import BASE_DIR, MODEL_DIR, DEVICE, AGE_GROUPS
from model import build_model

def predict_age(img_path):
    if not os.path.exists(img_path):
        print(f" Không tìm thấy ảnh: {img_path}")
        return
    
    img = cv2.imread(img_path)
    if img is None:
        print("Không đọc được ảnh")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        print("⚠️ Không phát hiện khuôn mặt! Dùng toàn bộ ảnh gốc để predict.")
        face_img = img
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.2 * max(w, h))
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        face_img = img[y1:y2, x1:x2]
        print(f"✅ Phát hiện khuôn mặt lớn nhất ({w}x{h} px).")

    DEBUG_DIR = os.path.join(BASE_DIR, "test")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_crop_path = os.path.join(DEBUG_DIR, "anh_duoc_cat.png")
    cv2.imwrite(debug_crop_path, face_img)
    print(f"📸 Đã lưu ảnh khuôn mặt cắt được vào: {debug_crop_path}")

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_img)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

    model_path = os.path.join(MODEL_DIR, "best_resnet50_utkface.pth")
    if not os.path.exists(model_path):
        print(f" Không tìm thấy model tại {model_path}")
        return

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        class_idx = np.argmax(probs)
        confidence = probs[class_idx] * 100

    group_name = AGE_GROUPS[class_idx]
    print(f"\nNhóm tuổi dự đoán: {group_name}")
    print(f"Độ tin cậy: {confidence:.1f}%")
    print("Xác suất các nhóm:")
    for i, prob in enumerate(probs):
        print(f"  {AGE_GROUPS[i]}: {prob*100:.1f}%")

if __name__ == "__main__":
    # Thay đổi đường dẫn đến ảnh cần test ở đây
    test_img = os.path.join(BASE_DIR, "test", "5tuoi.jpg")
    predict_age(test_img)