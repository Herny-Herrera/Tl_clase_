import matplotlib.pyplot as plt
import json
import os

# Cargar historial de entrenamiento
results_path = "results/"
history_path = os.path.join(results_path, "history.json")

if os.path.exists(history_path):
    with open(history_path, "r") as f:
        history = json.load(f)

    # Graficar la pérdida y la precisión
    plt.figure(figsize=(12, 4))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Evolución de la Pérdida')

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Evolución de la Precisión')

    plt.show()
else:
    print("No se encontró historial de entrenamiento.")
