import os
import numpy as np
import matplotlib.pyplot as plt

from dataset_loader import load_dataset, AgeDataGenerator, AGE_GROUPS, NUM_CLASSES
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Đường dẫn tuyệt đối dựa trên vị trí file evaluate.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "UTKFace")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_resnet_age_model.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

X,y = load_dataset(DATASET_PATH)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = load_model(MODEL_PATH)

test_gen = AgeDataGenerator(X_test, y_test, batch_size=32, shuffle=False)
loss, accuracy = model.evaluate(test_gen)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

print("Đang tạo dự đoán để vẽ biểu đồ...")
preds = model.predict(test_gen)
pred_classes = np.argmax(preds, axis=1)

# Tên nhóm tuổi (ngắn gọn cho biểu đồ)
group_names = [AGE_GROUPS[i] for i in range(NUM_CLASSES)]

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, pred_classes, target_names=group_names))

# 1. Vẽ Confusion Matrix
cm = confusion_matrix(y_test, pred_classes)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=group_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "evaluate_confusion_matrix.png"))
plt.close()

# 2. Vẽ biểu đồ phân phối dự đoán đúng/sai theo nhóm tuổi
correct = (pred_classes == y_test)
correct_counts = [np.sum(correct[y_test == i]) for i in range(NUM_CLASSES)]
total_counts = [np.sum(y_test == i) for i in range(NUM_CLASSES)]
wrong_counts = [total_counts[i] - correct_counts[i] for i in range(NUM_CLASSES)]

x_pos = np.arange(NUM_CLASSES)
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x_pos - width/2, correct_counts, width, label='Đúng', color='#2ca02c')
bars2 = ax.bar(x_pos + width/2, wrong_counts, width, label='Sai', color='#d62728')
ax.set_xlabel('Nhóm tuổi')
ax.set_ylabel('Số lượng ảnh')
ax.set_title('Phân phối dự đoán Đúng/Sai theo nhóm tuổi')
ax.set_xticks(x_pos)
ax.set_xticklabels(group_names, rotation=30, ha='right')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "evaluate_accuracy_per_group.png"))
plt.close()

# 3. Vẽ đường cong ROC và tính AUC (Multi-class)
print("Đang tính toán và vẽ đường cong ROC (Multi-class)...")
# Binarize nhãn thực tế
y_test_bin = label_binarize(y_test, classes=range(NUM_CLASSES))

fpr = dict()
tpr = dict()
roc_auc = dict()

# Tính ROC và AUC cho từng nhóm tuổi
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Vẽ đồ thị ROC
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
for i, color in zip(range(NUM_CLASSES), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC {group_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2) # Đường chéo ngẫu nhiên
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Đường cong ROC và AUC cho 6 Nhóm tuổi')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(RESULTS_DIR, "evaluate_roc_curve.png"))
plt.close()

print(f"Đã lưu các biểu đồ phân tích (bao gồm ROC/AUC) vào thư mục: {RESULTS_DIR}")