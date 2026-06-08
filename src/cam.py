import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import BASE_DIR, MODEL_DIR, DEVICE, AGE_GROUPS
from model import build_model

def run_cam():
    MODEL_PATH = os.path.join(MODEL_DIR, "best_resnet50_utkface.pth")
    if not os.path.exists(MODEL_PATH):
        print(f"Không tìm thấy model tại {MODEL_PATH}")
        return

    print("Đang tải model...")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Đã tải model thành công!")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Lỗi: Không thể mở camera!")
        sys.exit(1)

    print("Nhấn ESC để thoát.")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ camera!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                pad = int(0.2 * max(w, h))
                y1 = max(0, y - pad)
                y2 = min(frame.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(frame.shape[1], x + w + pad)
                face_img = frame[y1:y2, x1:x2]

                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)
                
                input_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    class_idx = np.argmax(probs)
                    confidence = probs[class_idx] * 100
                    group_name = AGE_GROUPS[class_idx]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{group_name} ({confidence:.0f}%)"
                label_y = y - 10 if y - 10 > 10 else y + h + 25
                cv2.putText(frame, label, (x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Khong phat hien khuon mat",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        cv2.imshow("Age Classification", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_cam()