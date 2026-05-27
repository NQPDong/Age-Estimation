import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Định nghĩa các nhóm tuổi
AGE_GROUPS = {
    0: "0-2 Em bé",
    1: "3-12 Trẻ em",
    2: "13-18 Vị thành niên",
    3: "19-35 Thanh niên",
    4: "36-55 Trung niên",
    5: "55+ Người cao tuổi",
}

NUM_CLASSES = len(AGE_GROUPS)

def age_to_group(age):
    """Chuyển tuổi thành chỉ số nhóm tuổi (0-5)."""
    if age <= 2:
        return 0
    elif age <= 12:
        return 1
    elif age <= 18:
        return 2
    elif age <= 35:
        return 3
    elif age <= 55:
        return 4
    else:
        return 5

class AgeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=224, shuffle=True, augment=False):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = self.image_paths[batch_indices]
        batch_labels = self.labels[batch_indices]

        X, y = [], []
        for path, label in zip(batch_paths, batch_labels):
            img = cv2.imread(path)
            if img is not None:
                # 1. Chuyển hệ màu chuẩn
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                # 2. Resize
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                # Data Augmentation (Tăng cường dữ liệu)
                if self.augment:
                    # Lật ngang ngẫu nhiên
                    if np.random.rand() > 0.5:
                        img = cv2.flip(img, 1)
                    
                    # Thay đổi độ sáng ngẫu nhiên
                    if np.random.rand() > 0.5:
                        factor = np.random.uniform(0.7, 1.3)
                        img = np.clip(img * factor, 0, 255).astype(np.uint8)

                    # Xoay ảnh ngẫu nhiên (-15 đến 15 độ)
                    if np.random.rand() > 0.5:
                        angle = np.random.uniform(-15, 15)
                        h, w = img.shape[:2]
                        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

                    # Zoom ngẫu nhiên (phóng to/thu nhỏ 0.85x - 1.15x)
                    if np.random.rand() > 0.5:
                        scale = np.random.uniform(0.85, 1.15)
                        h, w = img.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        img = cv2.resize(img, (new_w, new_h))
                        # Cắt hoặc pad lại về kích thước gốc
                        if scale > 1.0:
                            # Cắt phần trung tâm
                            start_y = (new_h - h) // 2
                            start_x = (new_w - w) // 2
                            img = img[start_y:start_y + h, start_x:start_x + w]
                        else:
                            # Pad viền bằng reflect
                            pad_y = (h - new_h) // 2
                            pad_x = (w - new_w) // 2
                            img = cv2.copyMakeBorder(
                                img, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x,
                                cv2.BORDER_REFLECT_101
                            )
                # Dùng hàm preprocess_input riêng của ResNet50
                img = preprocess_input(img)
                
                X.append(img)
                y.append(label)
            
        return np.array(X, dtype="float32"), np.array(y, dtype="int32")

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_dataset(path):
    image_paths, labels = [], []
    for file in os.listdir(path):
        try:
            age = int(file.split('_')[0])
            img_path = os.path.join(path, file)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
                # Chuyển tuổi thành chỉ số nhóm tuổi (0-5)
                labels.append(age_to_group(age))
        except:
            continue
            
    return np.array(image_paths), np.array(labels)