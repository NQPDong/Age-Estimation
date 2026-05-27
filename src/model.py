from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers

NUM_CLASSES = 6  # 6 nhóm tuổi

def build_model():

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Mở khóa 50 layers cuối để học đặc trưng khuôn mặt tốt hơn 
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    # Đơn giản hóa Head và thêm L2 Regularizer để chống Overfitting
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Output softmax cho phân loại 6 nhóm tuổi
    output = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    return model