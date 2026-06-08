import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DATASET_PATH, CLEANED_PATH, MODEL_DIR, RESULTS_DIR, DEVICE, BATCH_SIZE, NUM_EPOCHS
from dataset_loader import filter_and_detect_faces, UTKFaceDataset, BalancedAgeSampler, get_transforms, show_augmented_images
from model import build_model

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'   [!] EarlyStopping: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_report.png'), dpi=300)
    plt.close()

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Thư mục chứa ảnh gốc không tồn tại: {DATASET_PATH}")
        return

    all_files = filter_and_detect_faces(DATASET_PATH, CLEANED_PATH)

    if len(all_files) == 0:
        print("Không tìm thấy dữ liệu ảnh khuôn mặt đạt chuẩn.")
        return

    # Chia dữ liệu 80/20 Train/Val
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

    train_transform, val_transform = get_transforms()
    train_dataset = UTKFaceDataset(train_files, transform=train_transform)
    val_dataset = UTKFaceDataset(val_files, transform=val_transform)

    # Hiển thị dữ liệu Augmentation
    aug_path = os.path.join(RESULTS_DIR, 'augmented_samples.png')
    show_augmented_images(train_dataset, save_path=aug_path)
    print(f"Đã lưu ảnh mẫu augment tại: {aug_path}")

    train_sampler = BalancedAgeSampler(train_dataset, batch_size=BATCH_SIZE)
    # Trên môi trường chung dùng num_workers=0 để tránh lỗi tiến trình trên Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
    early_stop_controller = EarlyStopping(patience=7, min_delta=0.001)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] [Train]")
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100. * train_correct / train_total:.2f}%"})

        epoch_train_loss = running_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] [Val]")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        scheduler.step()

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"\n[Epoch {epoch+1}] Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_resnet50_utkface.pth'))
            print(f"[*] Đã lưu Checkpoint (Val Loss: {best_val_loss:.4f})")

        early_stop_controller(epoch_val_loss)
        if early_stop_controller.early_stop:
            print(">>> Kích hoạt Early Stopping!")
            break

    plot_training_history(history)
    print("Huấn luyện hoàn tất!")

if __name__ == '__main__':
    main()