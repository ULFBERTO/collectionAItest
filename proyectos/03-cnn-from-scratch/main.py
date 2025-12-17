import os # Librería del sistema operativo. Usada aquí para configurar variables de entorno y solucionar conflictos.
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python' # Solución para error de compatibilidad con Protobuf.

import tensorflow as tf # Librería principal de IA. Contiene Keras para crear y entrenar la red neuronal.
import matplotlib.pyplot as plt # Librería para generar gráficas. La usaremos para visualizar la pérdida y precisión.
import math # Funciones matemáticas estándar. Usada para calcular pasos por época (redondeo hacia arriba).
import numpy as np # Librería de cálculo numérico. Maneja los arrays de datos (matrices de píxeles) de forma eficiente.

# 1. Configuración y Carga de Datos
print("Cargando dataset Fashion MNIST desde Keras...")
# Usamos tf.keras.datasets que viene incluido, evitando problemas de dependencias con tfds
(imagenes_entrenamiento, etiquetas_entrenamiento), (imagenes_prueba, etiquetas_prueba) = tf.keras.datasets.fashion_mnist.load_data()

# Nombres de las clases para convertir los números de etiqueta (0-9) a texto legible.
# El dataset Fashion MNIST usa números: 0=T-shirt/top, 1=Trouser, etc.
nombres_clases = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_ejemplos_entrenamiento = len(imagenes_entrenamiento)
num_ejemplos_pruebas = len(imagenes_prueba)

print(f"Ejemplos de entrenamiento: {num_ejemplos_entrenamiento}")
print(f"Ejemplos de prueba: {num_ejemplos_pruebas}")
print(f"Clases: {nombres_clases}")

# 2. Preprocesamiento (Normalización y Reshape)
# Normalizar a 0-1
# Los píxeles van de 0 (negro) a 255 (blanco).
# Dividir por 255.0 convierte estos valores al rango 0-1, lo cual ayuda a la red neuronal a aprender más rápido y mejor.
imagenes_entrenamiento = imagenes_entrenamiento / 255.0
imagenes_prueba = imagenes_prueba / 255.0

# Reshape para incluir el canal de color (28, 28, 1) necesario para Conv2D
# Las redes CNN esperan datos en 4 dimensiones: (Lote, Alto, Ancho, Canales)
# -1: "Calcula tú el tamaño". Mantiene el número total de imágenes (60,000).
# 28, 28: Alto y Ancho de la imagen.
# 1: Canal de color. 1 para Blanco y Negro (Grayscale), 3 sería para RGB.
imagenes_entrenamiento = imagenes_entrenamiento.reshape(-1, 28, 28, 1)
imagenes_prueba = imagenes_prueba.reshape(-1, 28, 28, 1)

# 3. Definición del Modelo CNN
print("\nDefiniendo el modelo CNN...")
modelo = tf.keras.Sequential([
  # --- PARTE 1: EXTRACCIÓN DE CARACTERÍSTICAS (Ojos de la red) ---
  # Capa Convolucional 1:
  # - 32 filtros: ¿Por qué 32? Es un estándar para empezar. Significa que aprenderá 32 patrones distintos (un filtro para bordes verticales, otro para horizontales, curvas, etc.).
  # - kernel (3,3): Es el tamaño de la matriz de escaneo. 3x3 es ideal para captar detalles pequeños sin ser muy costoso computacionalmente.
  # - input_shape=(28, 28, 1): OBLIGATORIO. Debemos decirle a la red cuánto miden las fotos que entran (28 alto, 28 ancho, 1 color).
  # - activation='relu': "Rectified Linear Unit". Es la función matemática que decide si la neurona se activa.
  #   Básicamente convierte los negativos a cero (si no hay borde, apágate) y deja pasar los positivos (si hay borde, actívate).
  #   Es fundamental para que la red aprenda patrones no lineales.
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  
  # Capa Pooling 1:
  # - Reduce la imagen a la mitad (de 28x28 a 14x14) quedándose con el valor más alto de cada cuadro 2x2.
  # - Ayuda a que la red sea más rápida y se centre en lo importante.
  tf.keras.layers.MaxPooling2D(2, 2),

  # Capa Convolucional 2:
  # - 64 filtros: Ahora busca características más complejas (combinación de líneas, formas simples).
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  
  # Capa Pooling 2:
  # - Reduce otra vez a la mitad (de 14x14 teóricos a 5x5 aprox en la práctica tras convoluciones).
  tf.keras.layers.MaxPooling2D(2,2),

  # --- PARTE 2: CLASIFICACIÓN (Cerebro de la red) ---
  # Aplanar (Flatten):
  # - ¿Por qué 3D? Porque al pasar por las capas Conv2D, la imagen gana "profundidad".
  #   Empezó siendo 28x28x1 (1 hoja de papel).
  #   Ahora es un "cubo" de 5x5x64 (64 hojas de papel pequeñas apiladas, una por cada filtro).
  # - Flatten aplasta ese cubo de 5x5x64 = 1600 píxeles totales en una sola fila larga.
  tf.keras.layers.Flatten(),
  
  # Capa Densa (Oculta):
  # - 100 neuronas: NO es obligatorio que sean 100. Es un número arbitrario (hiperparámetro).
  #   - Si pones muy pocas (ej: 10), la red no tendrá suficiente "memoria" para aprender patrones complejos.
  #   - Si pones demasiadas (ej: 1000), será lenta y podría memorizar los datos en vez de aprender (overfitting).
  #   - 100 es un buen punto medio para este problema.
  tf.keras.layers.Dense(100, activation='relu'),
  
  # Capa de Salida:
  # - 10 neuronas: Una por cada clase de ropa (Camiseta, Pantalón, etc.).
  # - softmax: Convierte los números en probabilidades (ej: 80% Camiseta, 5% Pantalón...).
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
# Aquí configuramos CÓMO va a aprender la red.
modelo.compile(
    # optimizer='adam': Es el algoritmo matemático que ajusta los pesos.
    # 'Adam' es el mejor "todoterreno" hoy en día. Ajusta la velocidad de aprendizaje automáticamente.
    # Otros optimizadores populares:
    # - SGD (Stochastic Gradient Descent): El clásico. Es más lento y requiere ajustar manualmante la velocidad, pero a veces logra soluciones más estables.
    # - RMSprop: Similar a Adam. Fue muy popular antes de Adam, especialmente para redes recurrentes (texto).
    # - Adagrad: Adapta la velocidad para cada parámetro. Útil cuando los datos son "dispersos" (tienen muchos ceros).
    optimizer='adam',
    
    # loss: Función de pérdida. Mide qué tan mal lo hizo la red. Queremos minimizar esto.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    
    # metrics=['accuracy']: Queremos monitorear la "Precisión" (porcentaje de aciertos) durante el entrenamiento.
    # Es más fácil de entender para humanos que el valor de 'loss'.
    metrics=['accuracy']
)

# Imprime un resumen de la arquitectura en la consola (la tabla que viste antes).
# Útil para verificar que las capas y tamaños son los que esperabas.
modelo.summary()

# 4. Entrenamiento
BATCH_SIZE = 32
print("\nIniciando entrenamiento...")
historial = modelo.fit(
    imagenes_entrenamiento, 
    etiquetas_entrenamiento,
    epochs=5,
    batch_size=BATCH_SIZE,
    validation_data=(imagenes_prueba, etiquetas_prueba)
)

# 5. Visualización de Resultados
print("\nGenerando gráficas de pérdida...")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"], label='Pérdida Entrenamiento')
plt.plot(historial.history["val_loss"], label='Pérdida Validación')
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel("# Epoca")
plt.ylabel("Precisión")
plt.plot(historial.history["accuracy"], label='Precisión Entrenamiento')
plt.plot(historial.history["val_accuracy"], label='Precisión Validación')
plt.legend()

# plt.show() # Comentado para que el script no se detenga aquí y pueda guardar el modelo automáticamente.

# 6. Evaluación final
print("\nEvaluando en set de pruebas...")
test_loss, test_accuracy = modelo.evaluate(imagenes_prueba, etiquetas_prueba)
print(f"Precisión en pruebas: {test_accuracy*100:.2f}%")

# 7. Guardar el modelo
print("\nGuardando el modelo...")
modelo.save('modelo_ropa.h5')
print("Modelo guardado como 'modelo_ropa.h5'")
