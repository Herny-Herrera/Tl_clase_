import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from Modelo import build_model
import os

# Cargar los datos
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalización
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Construir el modelo
input_shape = x_train.shape[1:]
model = build_model(input_shape, num_classes=10)

# Definir Callbacks
model_dir = "models/"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "alexnet_cifar10.h5")

checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint, early_stopping])
import json
import os

# Guardar historial de entrenamiento
history_dict = history.history
results_dir = "results/"
os.makedirs(results_dir, exist_ok=True)

history_path = os.path.join(results_dir, "history.json")

with open(history_path, "w") as f:
    json.dump(history_dict, f)

print(f"Historial de entrenamiento guardado en {history_path}")


# Guardar modelo final
model.save(os.path.join(model_dir, "final_model.h5"))
print(f"Modelo guardado en {model_dir}")
