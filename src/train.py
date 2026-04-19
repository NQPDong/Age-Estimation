import os

from dataset_loader import load_dataset, AgeDataGenerator
from model import build_model

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt

# Đường dẫn tuyệt đối 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "UTKFace")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

print("Loading dataset...")

X,y = load_dataset(DATASET_PATH)

print("Dataset shape:",X.shape)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = build_model()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=Huber(),
    metrics=['mae']
)

model.summary()

train_gen = AgeDataGenerator(X_train, y_train, batch_size=32)
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
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# save final model
model.save(os.path.join(MODEL_DIR, "last_resnet_age_model.h5"))

# plot training curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["train","validation"])

plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"))
plt.show()