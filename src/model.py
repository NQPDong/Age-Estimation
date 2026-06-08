import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Đóng băng các layer đầu, chỉ mở khóa layer3, layer4
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model