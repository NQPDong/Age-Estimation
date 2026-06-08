import os
import random
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from config import DATASET_PATH, CLEANED_PATH, AGE_GROUPS

def age_to_group(age):
    if age < 18:
        return 0
    elif 18 <= age <= 55:
        return 1
    else:
        return 2

def filter_and_detect_faces(source_dir, output_dir):
    """
    Dùng Haar Cascade để detect face. Chỉ lưu ảnh có face.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_image_paths = []
    all_files = []

    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))

    print(f"[!] Bắt đầu quét và detect khuôn mặt cho {len(all_files)} ảnh...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    import tqdm
    for img_path in tqdm.tqdm(all_files, desc="Đang xác thực và detect face"):
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)

        if os.path.exists(save_path):
            valid_image_paths.append(save_path)
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            cv2.imwrite(save_path, img)
            valid_image_paths.append(save_path)

    print(f"[✓] Hoàn tất! Giữ lại {len(valid_image_paths)} / {len(all_files)} ảnh đạt chuẩn (có face).")
    return valid_image_paths

class UTKFaceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.group_indices = {0: [], 1: [], 2: []}
        self.labels = []

        for i, path in enumerate(image_paths):
            filename = os.path.basename(path)
            try:
                age = int(filename.split('_')[0])
                label = age_to_group(age)
                self.labels.append(label)
                self.group_indices[label].append(i)
            except Exception:
                continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

class BalancedAgeSampler(Sampler):
    def __init__(self, dataset, batch_size=30):
        self.indices_0 = dataset.group_indices[0]
        self.indices_1 = dataset.group_indices[1]
        self.indices_2 = dataset.group_indices[2]
        self.batch_size = batch_size
        self.samples_per_group = batch_size // 3
        
        max_samples = max(len(self.indices_0), len(self.indices_1), len(self.indices_2))
        self.num_batches = max_samples // self.samples_per_group

    def __iter__(self):
        g0 = np.random.choice(self.indices_0, size=self.num_batches * self.samples_per_group, replace=True)
        g1 = np.random.choice(self.indices_1, size=self.num_batches * self.samples_per_group, replace=True)
        g2 = np.random.choice(self.indices_2, size=self.num_batches * self.samples_per_group, replace=True)

        all_indices = []
        for i in range(self.num_batches):
            batch = []
            start_idx = i * self.samples_per_group
            end_idx = (i + 1) * self.samples_per_group
            
            batch.extend(g0[start_idx:end_idx])
            batch.extend(g1[start_idx:end_idx])
            batch.extend(g2[start_idx:end_idx])
            
            # Xáo trộn các phần tử bên trong 1 batch để model không học theo thứ tự cố định
            random.shuffle(batch)
            all_indices.extend(batch)
            
        return iter(all_indices)

    def __len__(self):
        return self.num_batches * self.batch_size

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def show_augmented_images(dataset, num_images=6, save_path=None):
    import matplotlib.pyplot as plt
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    actual_num_images = min(num_images, len(dataset))
    if actual_num_images == 0:
        return
    indices = random.sample(range(len(dataset)), actual_num_images)
    
    for i, idx in enumerate(indices):
        if i >= actual_num_images: break
        img_tensor, label = dataset[idx]
        img = img_tensor.numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(AGE_GROUPS[label])
        axes[i].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Đoạn test logic này chỉ chạy khi thực thi trực tiếp script dataset_loader.py
    print(f"Đường dẫn dataset gốc: {DATASET_PATH}")
    if os.path.exists(DATASET_PATH):
        # Chạy lọc và crop face thử nghiệm
        print("Đang chạy thử nghiệm hàm Face Detection...")
        valid_paths = filter_and_detect_faces(DATASET_PATH, CLEANED_PATH)
        
        if len(valid_paths) > 0:
            print("\nĐang thử nghiệm tạo Dataset và Augmentation...")
            train_trans, _ = get_transforms()
            test_dataset = UTKFaceDataset(valid_paths, transform=train_trans)
            
            # Khởi tạo thư mục results để lưu file ảnh augment demo
            from config import RESULTS_DIR
            os.makedirs(RESULTS_DIR, exist_ok=True)
            demo_path = os.path.join(RESULTS_DIR, "demo_augmentation.png")
            
            show_augmented_images(test_dataset, num_images=6, save_path=demo_path)
            print(f"[Thành công] Đã xuất thử nghiệm ảnh augmentation tại: {demo_path}")
            
            # Test thử sampler xem có load đều không
            print("\nĐang kiểm tra tính cân bằng của Sampler...")
            from torch.utils.data import DataLoader
            sampler = BalancedAgeSampler(test_dataset, batch_size=30)
            loader = DataLoader(test_dataset, batch_size=30, sampler=sampler)
            
            for i, (images, labels) in enumerate(loader):
                if i >= 3: # Chỉ kiểm tra 3 batch đầu tiên
                    break
                
                counts = {0: 0, 1: 0, 2: 0}
                for lbl in labels.numpy():
                    counts[lbl] += 1
                print(f"Batch {i+1} - Tổng số mẫu: {len(labels)}")
                print(f"  Số lượng Nhóm 0 (<18): {counts[0]}")
                print(f"  Số lượng Nhóm 1 (18-55): {counts[1]}")
                print(f"  Số lượng Nhóm 2 (>55): {counts[2]}")
                print("  => " + ("Cân bằng chuẩn!" if counts[0] == counts[1] == counts[2] else "Bị lệch!"))
        else:
            print("Không có ảnh nào chứa khuôn mặt được tìm thấy.")
    else:
        print("Không tìm thấy dữ liệu ảnh. Vui lòng cho dữ liệu ảnh UTKFace vào thư mục dataset/UTKFace")