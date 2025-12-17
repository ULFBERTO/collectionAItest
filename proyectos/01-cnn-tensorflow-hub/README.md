# 🖼️ Proyecto: Red Convolucional con TensorFlow Hub

## 📋 Objetivo

Crear un clasificador de imágenes usando una red convolucional preentrenada de TensorFlow Hub y aplicar transfer learning para adaptarla a un dataset personalizado.

## 🎯 Lo que aprenderás

- ✅ Qué es una red convolucional (CNN)
- ✅ Cómo funcionan las capas convolucionales
- ✅ Transfer learning (reutilizar modelos preentrenados)
- ✅ TensorFlow Hub y modelos preentrenados
- ✅ Data augmentation (aumento de datos)
- ✅ Fine-tuning de modelos

## 🔧 Requisitos

```bash
pip install tensorflow tensorflow-hub matplotlib numpy pillow
```

## 📁 Estructura del Proyecto

```
01-cnn-tensorflow-hub/
├── README.md                    # Este archivo
├── 01-conceptos-cnn.md          # Teoría de CNNs
├── 02-basico-cifar10.py         # Ejemplo básico con CIFAR-10
├── 03-transfer-learning.py      # Transfer learning con TF Hub
├── 04-custom-dataset.py         # Clasificador personalizado
├── datos/                       # Dataset personalizado
│   ├── entrenamiento/
│   └── validacion/
└── modelos/                     # Modelos guardados
```

## 🚀 Pasos del Proyecto

### Paso 1: Entender CNNs (Teoría)

Lee [`01-conceptos-cnn.md`](./01-conceptos-cnn.md) para entender:
- Qué son las capas convolucionales
- Cómo funcionan los filtros
- Pooling y reducción de dimensionalidad
- Arquitecturas famosas (VGG, ResNet, MobileNet)

### Paso 2: Ejemplo Básico

Ejecuta [`02-basico-cifar10.py`](./02-basico-cifar10.py):

```bash
python 02-basico-cifar10.py
```

Este script:
- Carga el dataset CIFAR-10 (10 clases de objetos)
- Crea una CNN simple desde cero
- Entrena el modelo
- Evalúa la precisión

**Tiempo estimado:** 15-20 minutos en CPU

### Paso 3: Transfer Learning

Ejecuta [`03-transfer-learning.py`](./03-transfer-learning.py):

```bash
python 03-transfer-learning.py
```

Este script:
- Carga un modelo preentrenado de TensorFlow Hub (MobileNetV2)
- Congela las capas base
- Añade capas personalizadas
- Entrena solo las capas nuevas
- Compara con el modelo desde cero

**Tiempo estimado:** 10 minutos en CPU

**Resultado esperado:** >90% de precisión (vs ~70% del modelo desde cero)

### Paso 4: Dataset Personalizado

Ejecuta [`04-custom-dataset.py`](./04-custom-dataset.py):

```bash
python 04-custom-dataset.py
```

Este script:
- Te guía para crear tu propio dataset
- Sugiere categorías (perros vs gatos, flores, objetos, etc.)
- Aplica data augmentation
- Entrena un clasificador personalizado
- Guarda el modelo para uso futuro

## 🎨 Ideas para Datasets Personalizados

1. **Clasificador de Frutas**: Manzanas, naranjas, plátanos
2. **Detector de Emociones**: Feliz, triste, neutral (usando caras)
3. **Clasificador de Vehículos**: Coche, moto, bicicleta
4. **Identificador de Mascotas**: Perro, gato, pájaro
5. **Clasificador de Ropa**: Camiseta, pantalón, zapatos

## 📊 Conceptos Clave

### ¿Qué es Transfer Learning?

En lugar de entrenar una CNN desde cero (que requiere millones de imágenes y días de entrenamiento), usamos un modelo ya entrenado en ImageNet (1.4 millones de imágenes, 1000 clases).

**Ventajas:**
- ✅ Entrena mucho más rápido
- ✅ Necesita menos datos
- ✅ Mejor precisión con menos recursos

**Proceso:**
```
Modelo Preentrenado (ImageNet)
        ↓
Congelar capas base
        ↓
Añadir capas personalizadas
        ↓
Entrenar solo las nuevas capas
        ↓
(Opcional) Fine-tune todo el modelo
```

### Modelos Disponibles en TensorFlow Hub

| Modelo | Tamaño | Precisión | Velocidad |
|--------|--------|-----------|-----------|
| MobileNetV2 | Pequeño | Alta | ⚡⚡⚡ Muy rápida |
| ResNet50 | Mediano | Muy alta | ⚡⚡ Rápida |
| EfficientNet | Variable | Excelente | ⚡⚡⚡ Muy rápida |
| InceptionV3 | Grande | Muy alta | ⚡ Media |

**Recomendación para empezar:** MobileNetV2

## 📈 Resultados Esperados

### Modelo desde cero (CNN simple)
- Precisión en CIFAR-10: ~70%
- Tiempo de entrenamiento: 15-20 min (CPU)

### Transfer Learning (MobileNetV2)
- Precisión en CIFAR-10: >90%
- Tiempo de entrenamiento: 5-10 min (CPU)

### Dataset personalizado (100 imágenes por clase)
- Precisión esperada: 85-95%
- Tiempo de entrenamiento: 5 min (CPU)

## 🐛 Solución de Problemas

### Error: "Out of Memory"
```python
# Reduce el batch_size
batch_size = 16  # en lugar de 32
```

### Error: "No module named tensorflow"
```bash
pip install --upgrade tensorflow
```

### Entrenamiento muy lento
- Reduce el tamaño de las imágenes
- Usa menos epochs
- Considera usar Google Colab (GPU gratis)

## 🎓 Siguientes Pasos

Después de completar este proyecto:

1. **Experimenta** con diferentes arquitecturas
2. **Prueba** otros datasets de TensorFlow Datasets
3. **Implementa** data augmentation avanzada
4. **Despliega** tu modelo en una aplicación web
5. **Explora** object detection (YOLO, SSD)

## 📚 Recursos Adicionales

- [TensorFlow Hub](https://tfhub.dev/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Visualizar CNNs](https://poloclub.github.io/cnn-explainer/)

## ✅ Checklist de Completación

- [ ] Entendí qué es una CNN y cómo funciona
- [ ] Ejecuté el ejemplo básico con CIFAR-10
- [ ] Probé transfer learning con TF Hub
- [ ] Creé un dataset personalizado
- [ ] Entrené un clasificador para mi dataset
- [ ] Guardé y probé el modelo entrenado
- [ ] Experimenté con diferentes hiperparámetros

---

**Tiempo total estimado:** 4-6 horas

¡Buena suerte! 🚀
