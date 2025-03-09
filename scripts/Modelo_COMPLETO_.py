import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import numpy as np

# Cargar dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalizar imágenes
x_train, x_test = x_train / 255.0, x_test / 255.0

# Definir la arquitectura de AlexNet adaptada para CIFAR-10
def alexnet_model():
    model = keras.Sequential([
        layers.Conv2D(96, (3,3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        layers.Conv2D(384, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 clases en CIFAR-10
    ])
    return model

# Compilar el modelo
model = alexnet_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Entrenar el modelo
epochs = 20  # Más épocas para mejorar precisión
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Precisión en test: {test_acc:.4f}")

# Guardar modelo
model.save("alexnet_cifar10.h5")

# Graficar pérdidas y precisión
plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()
