"""
CNN con TensorFlow Hub - Transfer Learning
===========================================

Este script demuestra cómo usar modelos preentrenados de TensorFlow Hub
para clasificar imágenes con muy poco código y alta precisión.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce warnings de TF

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

print("=" * 70)
print("CNN CON TENSORFLOW HUB - TRANSFER LEARNING")
print("=" * 70)

print(f"\n✅ TensorFlow versión: {tf.__version__}")
print(f"✅ TensorFlow Hub importado correctamente")

# ============================================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================================
print("\n1️⃣  CARGANDO DATASET CIFAR-10")
print("-" * 70)

# CIFAR-10: 60,000 imágenes de 32x32 en 10 clases
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Nombres de las clases
class_names = ['✈️ Avión', '🚗 Auto', '🐦 Pájaro', '🐱 Gato', '🦌 Ciervo',
               '🐕 Perro', '🐸 Rana', '🐴 Caballo', '🚢 Barco', '🚚 Camión']

print(f"Datos de entrenamiento: {x_train.shape}")
print(f"Datos de prueba: {x_test.shape}")
print(f"Clases: {len(class_names)}")

# Normalizar imágenes a rango [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Tomar un subset para entrenar más rápido (opcional)
# Para producción, usa todo el dataset
QUICK_MODE = True  # Cambia a False para usar todo el dataset

if QUICK_MODE:
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    print(f"\n⚡ MODO RÁPIDO: Usando solo {len(x_train)} imágenes de entrenamiento")

# ============================================================================
# 2. VISUALIZAR DATOS
# ============================================================================
print("\n2️⃣  VISUALIZACIÓN DE DATOS")
print("-" * 70)

plt.figure(figsize=(12, 6))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]], fontsize=9)
    plt.axis('off')

plt.suptitle('Ejemplos del Dataset CIFAR-10', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/proyectos/01-cnn-tensorflow-hub/01-dataset-ejemplos.png', dpi=100, bbox_inches='tight')
print("✅ Guardado: 01-dataset-ejemplos.png")

# ============================================================================
# 3. MODELO 1: CNN SIMPLE DESDE CERO
# ============================================================================
print("\n\n3️⃣  MODELO 1: CNN SIMPLE DESDE CERO")
print("-" * 70)

modelo_simple = models.Sequential([
    # Entrada: 32x32x3
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Aplanar y capas densas
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 clases
], name='CNN_Simple')

modelo_simple.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(modelo_simple.summary())

print("\n⏳ Entrenando modelo simple...")
print("💡 Esto tomará unos minutos. ¡Ten paciencia!")

history_simple = modelo_simple.fit(
    x_train, y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# Evaluar
test_loss_simple, test_acc_simple = modelo_simple.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Precisión en test (CNN simple): {test_acc_simple*100:.2f}%")

# ============================================================================
# 4. MODELO 2: TRANSFER LEARNING CON MOBILENET
# ============================================================================
print("\n\n4️⃣  MODELO 2: TRANSFER LEARNING CON MOBILENETV2")
print("-" * 70)

# MobileNetV2 espera imágenes de 224x224
# Necesitamos redimensionar nuestras imágenes
IMAGE_SIZE = 224

def resize_images(images, size=IMAGE_SIZE):
    """Redimensiona imágenes a 224x224 para MobileNet"""
    return tf.image.resize(images, (size, size)).numpy()

print(f"⏳ Redimensionando imágenes a {IMAGE_SIZE}x{IMAGE_SIZE}...")
x_train_resized = resize_images(x_train)
x_test_resized = resize_images(x_test)
print(f"✅ Forma nueva: {x_train_resized.shape}")

# Cargar MobileNetV2 preentrenado en ImageNet
print("\n⏳ Descargando MobileNetV2 desde TensorFlow Hub...")
print("💡 Primera vez puede tomar un minuto. Luego se cachea.")

# URL del modelo en TF Hub
MOBILENET_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# Crear modelo con transfer learning
modelo_transfer = models.Sequential([
    # Capa de entrada
    layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    
    # Modelo preentrenado (congelado)
    hub.KerasLayer(MOBILENET_URL, trainable=False),
    
    # Capas personalizadas
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
], name='MobileNet_Transfer')

modelo_transfer.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(modelo_transfer.summary())

print("\n⏳ Entrenando modelo con transfer learning...")
print("💡 Será mucho más rápido porque solo entrenamos las capas finales")

history_transfer = modelo_transfer.fit(
    x_train_resized, y_train,
    batch_size=32,
    epochs=5,  # Menos epochs porque converge más rápido
    validation_split=0.2,
    verbose=1
)

# Evaluar
test_loss_transfer, test_acc_transfer = modelo_transfer.evaluate(
    x_test_resized, y_test, verbose=0
)
print(f"\n✅ Precisión en test (Transfer Learning): {test_acc_transfer*100:.2f}%")

# ============================================================================
# 5. COMPARACIÓN DE MODELOS
# ============================================================================
print("\n\n5️⃣  COMPARACIÓN DE RESULTADOS")
print("-" * 70)

print("\n📊 RESULTADOS FINALES:")
print(f"  CNN Simple:          {test_acc_simple*100:.2f}%")
print(f"  Transfer Learning:   {test_acc_transfer*100:.2f}%")
print(f"  Mejora:              +{(test_acc_transfer - test_acc_simple)*100:.2f}%")

# Visualización de comparación
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Precisión del modelo simple
axes[0, 0].plot(history_simple.history['accuracy'], label='Entrenamiento', linewidth=2)
axes[0, 0].plot(history_simple.history['val_accuracy'], label='Validación', linewidth=2)
axes[0, 0].set_title('CNN Simple - Precisión', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('Precisión')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: Pérdida del modelo simple
axes[0, 1].plot(history_simple.history['loss'], label='Entrenamiento', linewidth=2)
axes[0, 1].plot(history_simple.history['val_loss'], label='Validación', linewidth=2)
axes[0, 1].set_title('CNN Simple - Pérdida', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('Pérdida')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Precisión transfer learning
axes[1, 0].plot(history_transfer.history['accuracy'], label='Entrenamiento', linewidth=2)
axes[1, 0].plot(history_transfer.history['val_accuracy'], label='Validación', linewidth=2)
axes[1, 0].set_title('Transfer Learning - Precisión', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Época')
axes[1, 0].set_ylabel('Precisión')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Comparación de barras
modelos = ['CNN\nSimple', 'Transfer\nLearning']
precisiones = [test_acc_simple * 100, test_acc_transfer * 100]
colores = ['#3498db', '#2ecc71']

bars = axes[1, 1].bar(modelos, precisiones, color=colores, edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Precisión (%)', fontsize=11)
axes[1, 1].set_title('Comparación Final', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim(0, 100)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Añadir valores encima de las barras
for bar, val in zip(bars, precisiones):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/proyectos/01-cnn-tensorflow-hub/02-comparacion-modelos.png', 
            dpi=100, bbox_inches='tight')
print("\n✅ Guardado: 02-comparacion-modelos.png")

# ============================================================================
# 6. PREDICCIONES DE EJEMPLO
# ============================================================================
print("\n\n6️⃣  PREDICCIONES DE EJEMPLO")
print("-" * 70)

# Seleccionar imágenes aleatorias
indices = np.random.choice(len(x_test), 12, replace=False)

# Hacer predicciones
predicciones_simple = modelo_simple.predict(x_test[indices], verbose=0)
predicciones_transfer = modelo_transfer.predict(
    resize_images(x_test[indices]), verbose=0
)

# Visualizar
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, idx in enumerate(indices):
    # Mostrar imagen
    axes[i].imshow(x_test[idx])
    
    # Predicción y etiqueta real
    pred_simple = np.argmax(predicciones_simple[i])
    pred_transfer = np.argmax(predicciones_transfer[i])
    real = y_test[idx][0]
    
    # Color: verde si acierta, rojo si falla
    color_simple = 'green' if pred_simple == real else 'red'
    color_transfer = 'blue' if pred_transfer == real else 'red'
    
    # Título
    titulo = f"Real: {class_names[real]}\n"
    titulo += f"CNN: {class_names[pred_simple]}\n"
    titulo += f"TL: {class_names[pred_transfer]}"
    
    axes[i].set_title(titulo, fontsize=9)
    axes[i].axis('off')

plt.suptitle('Predicciones: CNN Simple vs Transfer Learning', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/proyectos/01-cnn-tensorflow-hub/03-predicciones.png', 
            dpi=100, bbox_inches='tight')
print("✅ Guardado: 03-predicciones.png")

# ============================================================================
# 7. GUARDAR MODELO
# ============================================================================
print("\n\n7️⃣  GUARDANDO MODELO")
print("-" * 70)

# Guardar el mejor modelo
modelo_path = 'd:/EVIROMENT/PracticaIA/proyectos/01-cnn-tensorflow-hub/modelo_transfer_learning.h5'
modelo_transfer.save(modelo_path)
print(f"✅ Modelo guardado en: {modelo_path}")

print("\n💡 Para cargar el modelo más tarde:")
print("   modelo = tf.keras.models.load_model(modelo_path, custom_objects={'KerasLayer': hub.KerasLayer})")

# ============================================================================
# 8. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n\n" + "=" * 70)
print("🎉 ¡ENTRENAMIENTO COMPLETADO!")
print("=" * 70)

print("\n📚 LO QUE APRENDISTE:")
print("  ✅ Cargar y preparar datasets de imágenes")
print("  ✅ Crear una CNN desde cero")
print("  ✅ Usar modelos preentrenados con TensorFlow Hub")
print("  ✅ Transfer learning: aprovechar conocimiento existente")
print("  ✅ Comparar diferentes enfoques")
print("  ✅ Guardar y visualizar resultados")

print("\n🔍 CONCEPTOS CLAVE:")
print("  • Transfer Learning te da mejor precisión con menos datos y tiempo")
print("  • MobileNetV2 ya conoce patrones visuales básicos de ImageNet")
print("  • Solo entrenamos las capas finales para nuestras 10 clases")
print("  • Resultado: ~30-40% más de precisión que entrenar desde cero")

print("\n🚀 PRÓXIMOS PASOS:")
print("  1. Experimenta con otros modelos de TF Hub")
print("  2. Prueba con tu propio dataset de imágenes")
print("  3. Ajusta hiperparámetros (learning rate, epochs, etc.)")
print("  4. Implementa data augmentation")
print("  5. Fine-tune: descongelar algunas capas del modelo base")

print("\n📖 RECURSOS:")
print("  • TensorFlow Hub: https://tfhub.dev/")
print("  • Más datasets: https://www.tensorflow.org/datasets")

print("\n" + "=" * 70)

plt.show()
