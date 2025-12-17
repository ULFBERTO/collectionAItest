"""
Fundamentos de Álgebra Lineal para IA
======================================

Este archivo contiene ejemplos prácticos de operaciones con vectores y matrices.
Ejecuta cada sección y observa los resultados.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("VECTORES Y MATRICES - CONCEPTOS BÁSICOS")
print("=" * 60)

# ============================================================================
# 1. VECTORES
# ============================================================================
print("\n1️⃣  VECTORES")
print("-" * 60)

# Crear vectores
vector_simple = np.array([1, 2, 3, 4, 5])
print(f"Vector simple: {vector_simple}")
print(f"Dimensión: {vector_simple.shape}")

# Operaciones básicas
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"\nv1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {v1 + v2}")
print(f"v1 - v2 = {v1 - v2}")
print(f"v1 * 2 = {v1 * 2}")  # Multiplicación escalar
print(f"v1 * v2 = {v1 * v2}")  # Producto elemento por elemento

# Producto punto (muy importante en IA)
producto_punto = np.dot(v1, v2)
print(f"\nProducto punto v1 · v2 = {producto_punto}")
print("💡 El producto punto mide cuán 'alineados' están dos vectores")

# Magnitud (longitud) de un vector
magnitud = np.linalg.norm(v1)
print(f"\nMagnitud de v1 = {magnitud:.2f}")

# 💡 ¿De dónde sale ese 3.74?
# Es el Teorema de Pitágoras en 3D: raíz(x² + y² + z²)
# v1 = [1, 2, 3]
# Magnitud = raíz(1² + 2² + 3²) = raíz(1 + 4 + 9) = raíz(14) ≈ 3.74
print(f"Cálculo manual: √({v1[0]}² + {v1[1]}² + {v1[2]}²) = √{1**2 + 2**2 + 3**2} = {np.sqrt(14):.2f}")

# Normalización (hacer que la magnitud sea 1)
# Simplemente dividimos cada número del vector por su magnitud (3.74)
v1_normalizado = v1 / magnitud
print(f"\nv1 normalizado = {v1_normalizado}")
print(f"   -> [1/{magnitud:.2f}, 2/{magnitud:.2f}, 3/{magnitud:.2f}]")
print(f"Magnitud de v1 normalizado = {np.linalg.norm(v1_normalizado):.2f}")

# ============================================================================
# 2. MATRICES
# ============================================================================
print("\n\n2️⃣  MATRICES")
print("-" * 60)

# Crear matrices
matriz_2x3 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"Matriz 2x3:\n{matriz_2x3}")
print(f"Forma: {matriz_2x3.shape}")

# Matriz identidad (1s en diagonal, 0s en el resto)
identidad = np.eye(3)
print(f"\nMatriz identidad 3x3:\n{identidad}")

# Matriz de ceros
ceros = np.zeros((2, 4))
print(f"\nMatriz de ceros 2x4:\n{ceros}")

# Matriz de unos
unos = np.ones((3, 3))
print(f"\nMatriz de unos 3x3:\n{unos}")

# Matriz aleatoria (muy usada en IA para inicializar pesos)
aleatoria = np.random.randn(2, 3)  # Distribución normal
print(f"\nMatriz aleatoria 2x3:\n{aleatoria}")

# ============================================================================
# 3. OPERACIONES CON MATRICES
# ============================================================================
print("\n\n3️⃣  OPERACIONES CON MATRICES")
print("-" * 60)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matriz A:\n{A}")
print(f"\nMatriz B:\n{B}")

# Suma de matrices
print(f"\nA + B =\n{A + B}")

# Multiplicación elemento por elemento
print(f"\nA * B (elemento por elemento) =\n{A * B}")

# Multiplicación matricial (la importante)
# REGLA: "FILA contra COLUMNA"
# C[0,0] = Fila 0 de A · Columna 0 de B
# C[0,1] = Fila 0 de A · Columna 1 de B
# ... y así sucesivamente.

resultado = A @ B
print(f"\nA @ B (multiplicación matricial) =\n{resultado}")

print("\n💡 DESGLOSE DEL CÁLCULO (Fila x Columna):")
print(f"  [19] = (1*5) + (2*7)  = 5 + 14")
print(f"  [22] = (1*6) + (2*8)  = 6 + 16")
print(f"  [43] = (3*5) + (4*7)  = 15 + 28")
print(f"  [50] = (3*6) + (4*8)  = 18 + 32")
print("Esta operación combina información de todas las entradas con todos los pesos.")

# Transpuesta (Cambiar filas por columnas)
# Simplemente "giramos" la matriz.
# La primera fila se convierte en la primera columna.
print(f"\nA original:\n{A}")
print(f"A transpuesta (A.T):\n{A.T}")
print("💡 Fíjate: El [2] que estaba arriba a la derecha, ahora está abajo a la izquierda.")
print("   Las filas se volvieron columnas.")

# ============================================================================
# 4. APLICACIÓN: SIMILITUD ENTRE VECTORES
# ============================================================================
print("\n\n4️⃣  APLICACIÓN PRÁCTICA: SIMILITUD COSENO")
print("-" * 60)

def similitud_coseno(v1, v2):
    """
    Calcula la similitud coseno entre dos vectores.
    Retorna un valor entre -1 y 1.
    """
    # 1. Producto Punto (Numerador): Mide cuánto se "solapan" los vectores
    # np.dot multiplica elemento a elemento y suma los resultados
    dot_product = np.dot(v1, v2)
    
    # 2. Magnitudes (Denominador): El largo de cada vector
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    print(f"\n--- Calculando Similitud entre {v1} y {v2} ---")
    
    # Explicación visual del Producto Punto (np.dot)
    # Creamos una cadena de texto para mostrar: "(1*1) + (1*1) + (0*0)"
    terms = [f"({a}*{b})" for a, b in zip(v1, v2)]
    calculation = " + ".join(terms)
    
    print(f"1. Producto Punto (np.dot): {calculation} = {dot_product}")
    print(f"   (Multiplicamos parejas y sumamos todo)")
    print(f"2. Magnitudes: ||v1||={norm_v1:.2f}, ||v2||={norm_v2:.2f}")
    
    # Evitar división por cero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
        
    return dot_product / (norm_v1 * norm_v2)

# Ejemplo: representación de palabras como vectores
# (esto es básico, los embeddings reales son mucho más complejos)
perro = np.array([1, 1, 0])    # [animal, mascota, fruta]
gato = np.array([1, 1, 0])     # [animal, mascota, fruta]
manzana = np.array([0, 0, 1])  # [animal, mascota, fruta]

print("Representación simplificada de palabras:")
print(f"perro = {perro}")
print(f"gato = {gato}")
print(f"manzana = {manzana}")

print(f"\nSimilitud perro-gato: {similitud_coseno(perro, gato):.2f}")
print(f"Similitud perro-manzana: {similitud_coseno(perro, manzana):.2f}")
print("💡 Perro y gato son similares (1.0), manzana es diferente (0.0)")

# ============================================================================
# 4b. APLICACIÓN: SUMA PONDERADA (REDES NEURONALES Y DATA SCIENCE)
# ============================================================================
print("\n\n4️⃣b  APLICACIÓN PRÁCTICA: SUMA PONDERADA")
print("-" * 60)

# Ejemplo 1: Lista de compras (La forma más fácil de entenderlo)
# Imagina que tienes cantidades de productos y sus precios unitarios.
cantidades = np.array([2, 5, 3])    # [2 Manzanas, 5 Bananas, 3 Naranjas]
precios = np.array([0.5, 0.2, 0.8]) # [$0.50, $0.20, $0.80]

# El total es la suma de (cantidad * precio) para cada item.
# ¡Eso es exactamente el producto punto!
total_a_pagar = np.dot(cantidades, precios)

print(f"Cantidades: {cantidades}")
print(f"Precios: {precios}")
print(f"Total a pagar (Producto Punto): ${total_a_pagar:.2f}")
print("💡 Cálculo: (2*0.5) + (5*0.2) + (3*0.8) = 1.0 + 1.0 + 2.4 = 4.4")

# Ejemplo 2: Una Neurona Artificial (El cerebro de la IA)
# Una neurona toma varias entradas (inputs), les da una importancia (pesos),
# y decide si se activa o no.
inputs = np.array([1.0, 0.5, -1.0]) # Señales de entrada (ej. pixeles)
pesos = np.array([2.0, 1.5, 0.5])   # Importancia de cada señal (Weights)
bias = 0.1                          # Umbral de activación

# La operación fundamental de TODA red neuronal es esta:
activacion = np.dot(inputs, pesos) + bias

print(f"\nInputs de la neurona: {inputs}")
print(f"Pesos sinápticos: {pesos}")
print(f"Activación (Inputs · Pesos + Bias): {activacion:.2f}")
print("💡 Así es como las redes neuronales procesan información: suman entradas ponderadas.")

# ============================================================================
# 5. VISUALIZACIÓN
# ============================================================================
print("\n\n5️⃣  VISUALIZACIÓN DE VECTORES")
print("-" * 60)

# Crear figura con subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Vectores 2D
ax1 = axes[0]
vectores = np.array([[2, 3], [1, 4], [-2, 1]])
colores = ['red', 'blue', 'green']

ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-3, 3)
ax1.set_ylim(-1, 5)
ax1.set_aspect('equal')

for i, (vec, color) in enumerate(zip(vectores, colores)):
    ax1.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.2, 
              fc=color, ec=color, linewidth=2, label=f'v{i+1}')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Vectores en 2D')
ax1.legend()

# Subplot 2: Matriz de calor
ax2 = axes[1]
matriz_ejemplo = np.random.randn(5, 5)
im = ax2.imshow(matriz_ejemplo, cmap='coolwarm', aspect='auto')
ax2.set_title('Visualización de Matriz')
ax2.set_xlabel('Columnas')
ax2.set_ylabel('Filas')
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/01-fundamentos-matematicos/vectores_visualizacion.png', dpi=100)
print("✅ Gráfico guardado como 'vectores_visualizacion.png'")

# ============================================================================
# 6. EJERCICIOS
# ============================================================================
print("\n\n📝 EJERCICIOS PARA PRACTICAR")
print("-" * 60)
print("""
1. Crea dos vectores de 5 elementos y calcula:
   - Su suma
   - Su producto punto
   - Su similitud coseno

2. Crea una matriz 3x3 y:
   - Multiplícala por su transpuesta
   - Calcula su determinante (usa np.linalg.det)

3. Crea una función que normalice una matriz
   (cada fila debe tener magnitud 1)

4. Investiga qué es la matriz inversa y cómo calcularla
   (pista: np.linalg.inv)
""")

print("\n" + "=" * 60)
print("¡Has completado la lección de Vectores y Matrices!")
print("Siguiente: 02-calculo-gradientes.py")
print("=" * 60)
