README.md (Principal - En la raíz del repositorio)
md
Copiar
Editar
# 🧠 Proyecto: Entrenamiento de una CNN en CIFAR-10

Este proyecto implementa y entrena una **Red Neuronal Convolucional (CNN)** basada en **AlexNet** utilizando **TensorFlow/Keras** para la clasificación de imágenes del dataset **CIFAR-10**.

## 📂 **Estructura del Proyecto**
📦 TL_CLASE
┣ 📂 models (Modelos entrenados guardados)
┃ ┣ 📜 alexnet_cifar10.h5
┃ ┣ 📜 final_model.h5
┃ ┗ 📜 README.md
┣ 📂 results (Resultados del entrenamiento: historial, métricas, gráficos)
┃ ┣ 📜 history.json
┃ ┗ 📜 README.md
┣ 📂 scripts (Código fuente del proyecto)
┃ ┣ 📜 Entrenamiento.py (Entrena la red neuronal con CIFAR-10)
┃ ┣ 📜 Guardar_modelo_.py (Guarda el modelo entrenado)
┃ ┣ 📜 Modelo.py (Define la arquitectura de la CNN)
┃ ┗ 📜 resultados.py (Genera gráficos y evalúa el modelo)
┗ 📜 README.md (Este archivo)

## 🚀 **Requisitos**
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes librerías:

```bash
pip install tensorflow keras numpy matplotlib pandas
🏗️ Uso del Proyecto
1️⃣ Entrenar el modelo
Ejecuta el script de entrenamiento:


python scripts/Entrenamiento.py
2️⃣ Guardar el modelo
Si deseas guardar el modelo después del entrenamiento:


python scripts/Guardar_modelo_.py
3️⃣ Visualizar los resultados
Ejecuta el siguiente script para graficar las métricas de entrenamiento:


python scripts/resultados.py
📊 Resultados
El historial de entrenamiento se guarda en results/history.json y los modelos entrenados se encuentran en models/. Puedes visualizar el historial de entrenamiento con el siguiente código:


import json
import matplotlib.pyplot as plt

with open("results/history.json", "r") as f:
    history = json.load(f)

plt.plot(history["loss"], label="Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Evolución de la Pérdida durante el Entrenamiento")
plt.legend()
plt.show()
📜 Referencias
Documentación de TensorFlow
CIFAR-10 Dataset
