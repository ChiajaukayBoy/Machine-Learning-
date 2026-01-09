# Dataset para Clasificación de Residuos

## Estructura

Organiza tus imágenes en la siguiente estructura:

```
dataset/
├── train/
│   ├── plastic/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── paper/
│   │   ├── img001.jpg
│   │   └── ...
│   └── general/
│       ├── img001.jpg
│       └── ...
└── validation/
    ├── plastic/
    ├── paper/
    └── general/
```

## Recomendaciones

- **Cantidad mínima:** 100 imágenes por categoría
- **Cantidad óptima:** 500+ imágenes por categoría
- **Formatos:** JPG, JPEG, PNG
- **Split recomendado:** 70% train, 30% validation

## Datasets Públicos

Puedes usar estos datasets para comenzar:

1. **TrashNet**
   - URL: https://github.com/garythung/trashnet
   - 2527 imágenes de 6 categorías

2. **Waste Classification Dataset (Kaggle)**
   - URL: https://www.kaggle.com/datasets/techsash/waste-classification-data
   - 22k+ imágenes organizadas

3. **Garbage Classification (Kaggle)**
   - URL: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
   - 15k+ imágenes de 12 categorías

## Capturar tus propias imágenes

Usa el ESP32-S3 CAM web server para capturar imágenes propias:

1. Abre el web server del ESP32
2. Captura ~100 fotos de cada tipo de residuo
3. Descarga y organiza en las carpetas correspondientes
4. Asegúrate de variar:
   - Ángulos
   - Iluminación
   - Fondos
   - Condiciones (limpio, sucio, arrugado, etc.)

## Preprocesamiento

El script `train_model.py` se encarga automáticamente de:
- Redimensionar a 96x96 píxeles
- Normalizar valores de píxeles
- Data augmentation (rotación, flip, zoom, contraste)
