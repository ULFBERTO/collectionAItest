import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Configuración para evitar errores de protobuf (igual que en main.py)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# 1. Cargar el dataset (solo necesitamos las imágenes de prueba)
print("Cargando imágenes de prueba...")
(_, _), (imagenes_prueba, etiquetas_prueba) = tf.keras.datasets.fashion_mnist.load_data()

# Nombres de las clases
nombres_clases = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. Preprocesamiento (Normalizar y Reshape)
imagenes_prueba = imagenes_prueba / 255.0
imagenes_prueba = imagenes_prueba.reshape(-1, 28, 28, 1)

# 3. Cargar el modelo entrenado
print("Cargando el modelo...")
try:
    modelo = tf.keras.models.load_model('modelo_ropa.h5')
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de haber ejecutado main.py primero para crear el archivo 'modelo_ropa.h5'.")
    exit()

# 4. Seleccionar una imagen aleatoria
indice = random.randint(0, len(imagenes_prueba) - 1)
imagen = imagenes_prueba[indice]
etiqueta_real = etiquetas_prueba[indice]

# 5. Hacer la predicción
# El modelo espera un lote de imágenes, así que añadimos una dimensión extra
# De (28, 28, 1) pasa a (1, 28, 28, 1)
imagen_lote = np.expand_dims(imagen, axis=0)

prediccion = modelo.predict(imagen_lote)
indice_prediccion = np.argmax(prediccion[0]) # El índice con la probabilidad más alta
clase_predicha = nombres_clases[indice_prediccion]
clase_real = nombres_clases[etiqueta_real]

print(f"\n--- Resultado ---")
print(f"Real: {clase_real}")
print(f"Predicción: {clase_predicha}")

# 6. Mostrar la imagen
plt.figure(figsize=(5,5))
plt.imshow(imagen.reshape(28, 28), cmap=plt.cm.binary)
plt.title(f"Real: {clase_real} | Pred: {clase_predicha}")
plt.axis('off')
plt.show()
