import tensorflow as tf
import numpy as np

def guardar_modelo(model, filename):
    """Guarda el modelo entrenado en un archivo .h5"""
    model.save(filename)
    print(f"Modelo guardado en {filename}")

def cargar_modelo(filename):
    """Carga un modelo guardado"""
    model = tf.keras.models.load_model(filename)
    print(f"Modelo cargado desde {filename}")
    return model

def guardar_historial(history, filename):
    """Guarda el historial de entrenamiento"""
    np.save(filename, history.history)
    print(f"Historial guardado en {filename}")

def cargar_historial(filename):
    """Carga el historial de entrenamiento"""
    history = np.load(filename, allow_pickle=True).item()
    print(f"Historial cargado desde {filename}")
    return history
