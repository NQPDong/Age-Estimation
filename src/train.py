import os

from dataset_loader import load_dataset, AgeDataGenerator, AGE_GROUPS
from model import build_model

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn tuyệt đối 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "UTKFace")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

print("Loading dataset...")

X,y = load_dataset(DATASET_PATH)

print("Dataset shape:",X.shape)

# Thống kê phân phối nhóm tuổi
print("\n--- Phân phối nhóm tuổi ---")
for idx, name in AGE_GROUPS.items():
    count = np.sum(y == idx)
    print(f"  {name}: {count} ảnh ({count/len(y)*100:.1f}%)")
print()

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Giữ tỉ lệ nhóm tuổi cân bằng giữa train/test
)

# Tính class weights tự động để bù mất cân bằng dữ liệu
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights_array))
print("Class weights:", class_weight_dict)

model = build_model()

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

train_gen = AgeDataGenerator(X_train, y_train, batch_size=32, augment=True)
val_gen = AgeDataGenerator(X_test, y_test, batch_size=32, shuffle=False)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_resnet_age_model.h5"),
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict
)

# save final model
model.save(os.path.join(MODEL_DIR, "last_resnet_age_model.h5"))

# 1. Vẽ đồ thị Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title("Đường cong huấn luyện: Loss (CrossEntropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"))
plt.close()

# 2. Vẽ đồ thị Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#2ca02c', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#d62728', linewidth=2)
plt.title("Đường cong huấn luyện: Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(RESULTS_DIR, "training_accuracy.png"))
plt.close()

print(f"Đã lưu biểu đồ Loss và Accuracy tại thư mục: {RESULTS_DIR}")