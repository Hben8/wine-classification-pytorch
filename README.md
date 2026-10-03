# Wine Classification with a PyTorch Neural Network

Multiclass classification of the classic **Wine** dataset using a fully‑connected neural network implemented in **PyTorch**.

> **Author / Autor:** Hugo Benjumea Martínez · [github.com/Hben8](https://github.com/Hben8)

---

## 🇬🇧 English

### Overview
End‑to‑end PyTorch pipeline that classifies wine samples into their **3 cultivars** from **13 chemical features**:

- Custom **`Dataset`** and **`DataLoader`** (mini‑batches)
- Label encoding and **`StandardScaler`** normalization
- Fully‑connected network (`WineNet`: 13 → 64 → 32 → 3)
- **CrossEntropyLoss** and **SGD with momentum**
- Training loop over multiple epochs tracking **train/test accuracy** to monitor overfitting

### Data (included)
- `Wine.csv` — 13 features + the `Wine` class label

### Tech stack
Python · PyTorch · scikit‑learn · pandas · NumPy

### How to run
```bash
pip install torch scikit-learn pandas numpy
jupyter notebook wine_classification_pytorch.ipynb
```

---

## 🇪🇸 Español

### Descripción
Pipeline completo en PyTorch que clasifica muestras de vino en sus **3 variedades** a partir de **13 características químicas**:

- **`Dataset`** y **`DataLoader`** propios (mini‑batches)
- Codificación de etiquetas y normalización con **`StandardScaler`**
- Red totalmente conectada (`WineNet`: 13 → 64 → 32 → 3)
- **CrossEntropyLoss** y **SGD con momentum**
- Bucle de entrenamiento con seguimiento de **accuracy en train/test** para vigilar el sobreajuste

### Datos (incluidos)
- `Wine.csv` — 13 características + la etiqueta de clase `Wine`

### Tecnologías
Python · PyTorch · scikit‑learn · pandas · NumPy

### Cómo ejecutarlo
```bash
pip install torch scikit-learn pandas numpy
jupyter notebook wine_classification_pytorch.ipynb
```
