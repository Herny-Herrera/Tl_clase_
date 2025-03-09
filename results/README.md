# 📂 Results

Este directorio contiene los resultados del entrenamiento:

- **`history.json`**: Historial de entrenamiento con métricas de precisión y pérdida.
- **Gráficos de pérdida y precisión** *(si los generaste)*.

### 📌 Visualizar historial en Python:
```python
import json
import matplotlib.pyplot as plt

with open("results/history.json", "r") as f:
    history = json.load(f)

plt.plot(history["loss"], label="Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend()
plt.show()

