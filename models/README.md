# 📂 Models

Este directorio almacena los modelos entrenados guardados en formato HDF5.

- **`alexnet_cifar10.h5`**: Modelo AlexNet entrenado.
- **`final_model.h5`**: Última versión del modelo.
- **Formato recomendado:** Guardar modelos en formato `.keras` en lugar de `.h5`.

### 📌 Cargar un modelo entrenado en Python:
```python
from tensorflow.keras.models import load_model

modelo = load_model("models/alexnet_cifar10.h5")
modelo.summary()

