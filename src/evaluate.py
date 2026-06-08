import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import DATASET_PATH, CLEANED_PATH, MODEL_DIR, RESULTS_DIR, DEVICE, NUM_CLASSES, AGE_GROUPS
from dataset_loader import filter_and_detect_faces, UTKFaceDataset, get_transforms
from model import build_model

def evaluate_model():
    model_path = os.path.join(MODEL_DIR, 'best_resnet50_utkface.pth')
    if not os.path.exists(model_path):
        print("Không tìm thấy model checkpoint. Vui lòng train trước.")
        return

    all_files = filter_and_detect_faces(DATASET_PATH, CLEANED_PATH)
    _, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    _, val_transform = get_transforms()
    val_dataset = UTKFaceDataset(val_files, transform=val_transform)
    # Dùng num_workers=0 để tương thích tốt với Windows
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Đang đánh giá trên tập Validation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    group_names = [AGE_GROUPS[i] for i in range(NUM_CLASSES)]
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=group_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=group_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "evaluate_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    y_test_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {group_names[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "evaluate_roc_curve.png"))
    plt.close()

    print(f"Đã lưu các biểu đồ đánh giá tại: {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_model()