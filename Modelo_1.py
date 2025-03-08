import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def alexnet_model():
    model = keras.Sequential([
        layers.Conv2D(96, (3,3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(384, (3,3), activation='relu'),
        layers.Conv2D(384, (3,3), activation='relu'),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 clases en CIFAR-10
    ])
    return model
