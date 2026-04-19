import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

class AgeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=224, shuffle=True):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
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
                
                # 3. KẾT HỢP ƯU ĐIỂM CỦA NHÓM: Dùng hàm preprocess_input riêng của ResNet50
                # thay vì img / 255.0
                img = preprocess_input(img)
                
                X.append(img)
                y.append(label)
            
        # Đưa về kiểu float32 để đồng bộ và tính toán nhanh hơn
        return np.array(X, dtype="float32"), np.array(y, dtype="float32")

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
                
                # KẾT HỢP ƯU ĐIỂM CỦA NHÓM: Chia 116.0 (số tuổi cao nhất trong UTKFace)
                labels.append(age / 116.0) 
        except:
            continue
            
    return np.array(image_paths), np.array(labels)