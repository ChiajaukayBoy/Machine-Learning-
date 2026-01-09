# P3-EcoSort 🔄♻️

## Descripción del Proyecto

**EcoSort** es un sistema de clasificación automática de residuos utilizando técnicas de Machine Learning y Deep Learning. El proyecto implementa y compara tres enfoques diferentes para clasificar imágenes de residuos en tres categorías: **general**, **papel** y **plástico**.

### Objetivos

- Desarrollar modelos de clasificación robustos para residuos reciclables
- Comparar el rendimiento entre modelos clásicos de ML (Logistic Regression, SVM) y Deep Learning (CNN)
- Implementar un pipeline completo: EDA → Feature Engineering → Entrenamiento → Evaluación
- Generar visualizaciones profesionales en formato SVG con nomenclatura normalizada
- Manejar desbalanceo de clases mediante técnicas de data augmentation

---

## Estructura del Proyecto

```
P3-EcoSort/
├── data/
│   └── preprocessed/
│       ├── train/              # 2,020 imágenes de entrenamiento
│       │   ├── general/        # 838 imágenes (41.5%)
│       │   ├── paper/          # 797 imágenes (39.5%)
│       │   └── plastic/        # 385 imágenes (19.0%)
│       └── val/                # 507 imágenes de validación
│           ├── general/        # 210 imágenes (41.4%)
│           ├── paper/          # 201 imágenes (39.6%)
│           └── plastic/        # 96 imágenes (18.9%)
│
├── notebooks/
│   ├── 01_EDA.ipynb                      # Análisis Exploratorio de Datos
│   ├── 02_FEATURE_ENGINEERING.ipynb      # Extracción de características
│   ├── 03_ENTRENAMIENTO.ipynb            # Entrenamiento de modelos
│   └── 04_EVALUACION.ipynb               # Evaluación comparativa
│
├── result/
│   ├── features/              # Matrices de características (.npy)
│   ├── figures/               # Gráficos generados (.svg)
│   └── models/                # Modelos entrenados (.h5)
│
├── README.md
└── requirements.txt
```

---

## Dataset

### Características
- **Total de imágenes**: 2,527 (2,020 train + 507 val)
- **Dimensiones**: 512 × 384 píxeles (consistente en todo el dataset)
- **Formato**: JPEG
- **Clases**: 3 (general, paper, plastic)
- **Ratio de desbalanceo**: 2.18 (general:plastic)
- **Imágenes corruptas**: 0 (verificado)

### Distribución de Clases

| Clase    | Train | Val | Total | Proporción |
|----------|-------|-----|-------|------------|
| General  | 838   | 210 | 1,048 | 41.5%      |
| Paper    | 797   | 201 | 998   | 39.5%      |
| Plastic  | 385   | 96  | 481   | 19.0%      |

---

## Metodología

### 1. Análisis Exploratorio (01_EDA.ipynb)

**Objetivos**: Verificar integridad del dataset, analizar distribución de clases, detectar desbalanceo.

**Análisis realizados**:
- Conteo de imágenes por clase y split
- Verificación de dimensiones y formatos
- Cálculo de ratio de desbalanceo
- Detección de imágenes corruptas
- Visualización de muestras representativas

**Resultados clave**:
- Dataset consistente sin imágenes corruptas
- Desbalanceo moderado (ratio 2.18) requiere data augmentation
- Todas las imágenes tienen dimensiones uniformes

**Outputs**: 6 figuras SVG en `result/figures/` con prefijo `01_eda_`

---

### 2. Feature Engineering (02_FEATURE_ENGINEERING.ipynb)

**Objetivos**: Generar dataset balanceado mediante augmentation, extraer características para modelos clásicos.

#### Data Augmentation
Se implementaron **8 técnicas** para balancear el dataset:
- Rotación (±15°)
- Flip horizontal y vertical
- Ajuste de brillo (±0.2)
- Ajuste de contraste (±0.2)
- Zoom (0.9-1.1)
- Traslación (±10%)
- Ruido gaussiano
- Blur

**Meta**: Generar 10,000+ imágenes de entrenamiento balanceadas

#### Extracción de Características

**a) Características HOG (Histogram of Oriented Gradients)**
- Procesamiento en escala de grises
- Captura información de textura y bordes
- Dimensionalidad reducida para eficiencia

**b) Características de Color**
- Histogramas de 3 espacios de color: RGB, HSV, LAB
- Momentos de color (media, desviación estándar)
- Captura información cromática discriminativa

**c) Reducción de Dimensionalidad con PCA**
- Preservación del 95% de varianza explicada
- Optimización para GridSearchCV/RandomizedSearchCV
- Aceleración del entrenamiento

**d) Autoencoders para Representación No Supervisada**

**Autoencoder Convolucional Mejorado**:
- **Arquitectura**:
  ```
  Encoder:
    Conv2D(64) → BatchNorm → LeakyReLU → 64x64
    Conv2D(128) → BatchNorm → LeakyReLU → 32x32
    Conv2D(256) → BatchNorm → LeakyReLU → 16x16
    Conv2D(512) → BatchNorm → LeakyReLU → 8x8
    Flatten → FC(1024) → FC(latent_dim=980)
  
  Decoder:
    FC(latent_dim) → FC(1024) → Reshape(8x8x512)
    ConvTranspose2D(256) → BatchNorm → LeakyReLU → 16x16
    ConvTranspose2D(128) → BatchNorm → LeakyReLU → 32x32
    ConvTranspose2D(64) → BatchNorm → LeakyReLU → 64x64
    ConvTranspose2D(3) → Sigmoid → 128x128
  ```
- **Características clave**:
  - Latent space: 980 dimensiones (configurable)
  - Loss: MSE para reconstrucción
  
#### Validación
- Entrenamiento de Logistic Regression baseline
- Verificación de calidad de características extraídas

**Outputs**:
- Matrices `.npy` en `result/features/`:
  - `X_train_imagenes.npy` (imágenes aumentadas)
  - `X_val_imagenes.npy` (validación)
  - `features_train_pca.npy` (características PCA)
  - `features_val_pca.npy` (validación PCA)
  - `y_train.npy`, `y_val.npy` (etiquetas)

---

### 3. Entrenamiento (03_ENTRENAMIENTO.ipynb)

**Objetivos**: Entrenar 3 modelos con optimización de hiperparámetros, seleccionar mejores configuraciones.

#### Modelo 1: Logistic Regression (Baseline)
- **Tipo**: Modelo lineal clásico
- **Input**: Características PCA (HOG + color)
- **Optimización**: GridSearchCV (5-fold CV, n_jobs=1)
- **Hiperparámetros**:
  - `C`: [0.01, 0.1, 1, 10, 100]
  - `penalty`: ['l2']
  - `solver`: ['lbfgs', 'saga']
  - `multi_class`: ['multinomial']

#### Modelo 2: SVM (Advanced Classical)
- **Tipo**: Support Vector Machine
- **Input**: Características PCA
- **Optimización**: RandomizedSearchCV (30 iteraciones, 3-fold CV, n_jobs=1)
- **Hiperparámetros**:
  - `C`: [0.1, 1, 10, 100]
  - `kernel`: ['linear', 'rbf', 'poly']
  - `gamma`: ['scale', 'auto']
  - `degree`: [2, 3] (para kernel poly)

#### Modelo 3: CNN (Advanced Deep Learning)
- **Tipo**: Red Neuronal Convolucional
- **Input**: Imágenes crudas (512×384×3)
- **Framework**: PyTorch 2.6.0 con soporte CUDA
- **Arquitectura**:
  ```
  Conv Block 1: Conv2D(32) → ReLU → MaxPool → Dropout(0.25)
  Conv Block 2: Conv2D(64) → ReLU → MaxPool → Dropout(0.25)
  Conv Block 3: Conv2D(128) → ReLU → MaxPool → Dropout(0.4)
  Flatten
  Dense(512) → ReLU → Dropout(0.5)
  Dense(3) → Softmax
  ```
- **Training**:
  - Optimizer: Adam
  - Loss: CrossEntropyLoss
  - Device: CUDA (si disponible) / CPU
  - Epochs: Variable
  - Batch size: Configurable

#### Consideraciones Técnicas
⚠️ **Importante**: Se usa `n_jobs=1` en GridSearchCV y RandomizedSearchCV para evitar conflictos de serialización con PyTorch/CUDA en Windows. Esto garantiza compatibilidad a costa de velocidad de entrenamiento.

**Outputs**:
- Modelos `.h5` en `result/models/`:
  - `modelo_logistic_regression.h5`
  - `modelo_svm.h5`
  - `modelo_cnn.h5`
- Curvas de pérdida para CNN

---

### 4. Evaluación (04_EVALUACION.ipynb)

**Objetivos**: Comparar modelos, seleccionar el mejor, generar reporte técnico.

#### Métricas de Evaluación
- **F1 Macro** (métrica principal): Promedio no ponderado, sensible a clases minoritarias
- **F1 Micro**: Promedio ponderado por soporte
- **AUC-PR**: Área bajo curva Precision-Recall

#### Visualizaciones Generadas
- Comparación de métricas (gráfico de barras)
- Curvas de pérdida (solo CNN)
- Matrices de confusión normalizadas (formato decimal)
- Curvas Precision-Recall por clase
- Ranking de modelos

#### Criterios de Selección
Puntuación ponderada:
- F1 Macro: 50%
- F1 Micro: 30%
- AUC-PR: 20%

**Outputs**: Figuras SVG en `result/figures/` con prefijo `04_evaluacion_`

---

## Instalación

### Requisitos Previos
- Python 3.11+
- CUDA 12.4 (opcional, para aceleración GPU)
- 16GB RAM mínimo recomendado

### Instalación de Dependencias

1. **Crear entorno virtual** (recomendado):
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

2. **Instalar PyTorch con soporte CUDA**:
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

3. **Instalar demás dependencias**:
```bash
pip install -r requirements.txt
```

### Dependencias Principales
- **Deep Learning**: PyTorch 2.6.0, torchvision 0.21.0
- **Machine Learning**: scikit-learn 1.7.2, imbalanced-learn 0.14.0
- **Computer Vision**: OpenCV 4.12.0, albumentations 2.0.8
- **Data Processing**: NumPy 2.2.6, Pandas 2.3.3
- **Visualization**: Matplotlib 3.10.7, Seaborn 0.13.2, Plotly 6.5.0

---

## Uso

### Ejecución Completa del Pipeline

1. **Análisis Exploratorio**:
```bash
jupyter notebook notebooks/01_EDA.ipynb
```
Ejecutar todas las celdas para generar análisis del dataset.

2. **Feature Engineering**:
```bash
jupyter notebook notebooks/02_FEATURE_ENGINEERING.ipynb
```
Generar dataset aumentado y extraer características.

3. **Entrenamiento de Modelos**:
```bash
jupyter notebook notebooks/03_ENTRENAMIENTO.ipynb
```
Entrenar Logistic Regression, SVM y CNN.

4. **Evaluación Comparativa**:
```bash
jupyter notebook notebooks/04_EVALUACION.ipynb
```
Comparar modelos y seleccionar el mejor.

### Estructura de Ejecución
Los notebooks deben ejecutarse **secuencialmente** ya que cada uno depende de los outputs del anterior:
```
01_EDA → 02_FEATURE_ENGINEERING → 03_ENTRENAMIENTO → 04_EVALUACION
```

---

## Resultados Esperados

### Artefactos Generados

**Figuras** (`result/figures/`):
- `01_eda_*.svg`: Análisis exploratorio (6 figuras)
- `02_feature_*.svg`: Visualizaciones de features
- `04_evaluacion_*.svg`: Comparación de modelos y métricas

**Modelos** (`result/models/`):
- `modelo_logistic_regression.h5`
- `modelo_svm.h5`
- `modelo_cnn.h5`

**Features** (`result/features/`):
- `X_train_imagenes.npy`: Imágenes aumentadas
- `features_train_pca.npy`: Características PCA
- `y_train.npy`: Etiquetas de entrenamiento
- `X_val_imagenes.npy`: Imágenes de validación
- `features_val_pca.npy`: Características PCA validación
- `y_val.npy`: Etiquetas de validación

---

## Notas Técnicas

### Formato de Salida
- **Imágenes**: SVG (vectorial, alta calidad)
- **Nomenclatura**: `{numero_notebook}_{seccion}_{descripcion}.svg`
- **Matrices de confusión**: Formato decimal normalizado

### Optimización de Rendimiento
- **PCA**: Reduce dimensionalidad manteniendo 95% varianza
- **CUDA**: Aceleración GPU automática si disponible
- **Batch Processing**: Para CNNs y data augmentation
---

## Autor

Proyecto desarrollado como parte del curso de Machine Learning - Sexto Ciclo.

---

## Licencia

Este proyecto es de uso académico.
