import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import guardados  # Importamos funciones para cargar el modelo y el historial

# Cargar modelo entrenado
model = guardados.cargar_modelo("alexnet_cifar10.h5")

# Cargar datos de prueba
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0  # Normalizar imágenes

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Precisión en test: {test_acc:.4f}")

# Graficar historial de entrenamiento
history = guardados.cargar_historial("historial.npy")

plt.plot(history['accuracy'], label='Precisión entrenamiento')
plt.plot(history['val_accuracy'], label='Precisión validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
