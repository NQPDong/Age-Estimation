from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_model():
    input_layer = layers.Input(shape=(224,224,3))

    # Data Augmentation layer giúp đa dạng hoá dữ liệu
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    x = data_augmentation(input_layer)

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    # Đóng băng toàn bộ các layer trước để giữ đặc trưng ImageNet cơ bản
    for layer in base_model.layers:
        layer.trainable = False

    # Mở 30 lớp cuối cùng để model tự học lại chi tiết khuôn mặt
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    output = layers.Dense(1)(x)

    model = models.Model(inputs=input_layer, outputs=output)

    return model