import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "UTKFace")
CLEANED_PATH = os.path.join(BASE_DIR, "dataset", "UTKFace_Cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(CLEANED_PATH, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 30
NUM_EPOCHS = 15

# Nhóm tuổi b.md: <18, 18-55, >55
AGE_GROUPS = {
    0: "Dưới 18 tuổi",
    1: "18 đến 55 tuổi",
    2: "Trên 55 tuổi"
}
NUM_CLASSES = len(AGE_GROUPS)
