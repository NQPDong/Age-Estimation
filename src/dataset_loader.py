import os
import cv2
import numpy as np

def load_dataset(path, img_size=224, max_samples=None):

    images = []
    labels = []

    files = os.listdir(path)
    if max_samples:
        files = files[:max_samples]

    for file in files:
        try:
            # Lấy tuổi từ tên tệp tin (ví dụ: 25_0_1_...)
            age = int(file.split('_')[0])
            img_path = os.path.join(path, file)
            
            # Chỉ thêm vào nếu tệp tồn tại và là ảnh
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Tiền xử lý ảnh đồng bộ với file dự đoán (predict.py và cam.py) sử dụng không gian màu BGR
                    img = cv2.resize(img, (img_size, img_size))
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    # Chuẩn hóa tuổi về [0, 1]
                    labels.append(age / 100.0)
        except:
            continue

    return np.array(images), np.array(labels)
