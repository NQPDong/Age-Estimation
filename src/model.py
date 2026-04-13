from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_model():

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3)
    )

    # freeze convolution layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64,activation='relu')(x)

    output = layers.Dense(1)(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    return model