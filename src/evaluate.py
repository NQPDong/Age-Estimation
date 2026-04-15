import os

from dataset_loader import load_dataset, AgeDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Đường dẫn tuyệt đối dựa trên vị trí file evaluate.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "UTKFace")
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_model.h5")

X,y = load_dataset(DATASET_PATH)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model = load_model(MODEL_PATH)

test_gen = AgeDataGenerator(X_test, y_test, batch_size=32, shuffle=False)
loss,mae = model.evaluate(test_gen)

print("MAE:",mae)