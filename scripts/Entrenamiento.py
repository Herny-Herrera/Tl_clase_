import tensorflow as tf
from tensorflow import keras
import numpy as np
import modelo  # Importamos el modelo desde modelo.py
import guardados  # Para guardar el modelo entrenado

# Cargar dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar imágenes

# Crear y compilar el modelo
model = modelo.alexnet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
epochs = 1  # ePOCAS

history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# Guardar modelo entrenado
guardados.guardar_modelo(model, "alexnet_cifar10.h5")
guardados.guardar_historial(history, "historial.npy")
