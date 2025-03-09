from tensorflow.keras.models import load_model
import os

model_dir = "models/"
model_path = os.path.join(model_dir, "alexnet_cifar10.h5")

# Cargar modelo entrenado
model = load_model(model_path)
model.save(os.path.join(model_dir, "alexnet_cifar10_saved.h5"))
print("Modelo guardado exitosamente.")
