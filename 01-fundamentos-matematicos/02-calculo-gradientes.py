"""
Cálculo y Gradientes para IA
=============================

En este archivo aprenderás sobre derivadas y gradientes de forma intuitiva,
sin necesidad de recordar fórmulas complicadas.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("CÁLCULO Y GRADIENTES - SIN FÓRMULAS COMPLICADAS")
print("=" * 60)

# ============================================================================
# 1. ¿QUÉ ES UNA DERIVADA?
# ============================================================================
print("\n1️⃣  ¿QUÉ ES UNA DERIVADA?")
print("-" * 60)
print("""
💡 Explicación intuitiva:
La derivada te dice QUÉ TAN RÁPIDO cambia algo.

Ejemplo del mundo real:
- Posición: dónde estás
- Velocidad: derivada de la posición (cuán rápido cambias de lugar)
- Aceleración: derivada de la velocidad (cuán rápido cambia tu velocidad)

En IA:
- Función: tu error/pérdida
- Derivada: cuán rápido cambia el error cuando cambias los parámetros
""")

# Ejemplo visual
x = np.linspace(-5, 5, 100)
y = x**2  # Función cuadrática

# La derivada de x^2 es 2x
derivadas = 2 * x

plt.figure(figsize=(12, 5))

# Subplot 1: La función
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x²')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Función Cuadrática')
plt.legend()

# Marcar algunos puntos
puntos_x = [-3, 0, 3]
for px in puntos_x:
    py = px**2
    plt.plot(px, py, 'ro', markersize=10)
    plt.text(px, py + 2, f'({px}, {py})', ha='center')

# Subplot 2: La derivada
plt.subplot(1, 2, 2)
plt.plot(x, derivadas, 'r-', linewidth=2, label="f'(x) = 2x")
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title('Derivada (Pendiente)')
plt.legend()

for px in puntos_x:
    pd = 2 * px
    plt.plot(px, pd, 'ro', markersize=10)
    plt.text(px, pd + 1, f'pendiente = {pd}', ha='center')

plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/01-fundamentos-matematicos/derivadas_visualizacion.png', dpi=100)
print("✅ Gráfico guardado como 'derivadas_visualizacion.png'")

# ============================================================================
# 2. DERIVADAS NUMÉRICAS (APROXIMACIÓN)
# ============================================================================
print("\n\n2️⃣  CÁLCULO NUMÉRICO DE DERIVADAS")
print("-" * 60)

def derivada_numerica(f, x, h=1e-5):
    """
    Calcula la derivada de f en x usando diferencias finitas.
    
    Fórmula: f'(x) ≈ [f(x+h) - f(x)] / h
    
    No necesitas memorizarla, solo entender el concepto:
    "Cuánto cambia f cuando x cambia un poquito (h)"
    """
    return (f(x + h) - f(x)) / h

# Probemos con diferentes funciones
def f1(x):
    return x**2

def f2(x):
    return 2*x + 3

def f3(x):
    return np.sin(x)

print("Función f(x) = x²")
print(f"Derivada en x=3: {derivada_numerica(f1, 3):.4f}")
print(f"Derivada analítica: {2*3} (fórmula: 2x)")

print("\nFunción f(x) = 2x + 3")
print(f"Derivada en x=5: {derivada_numerica(f2, 5):.4f}")
print(f"Derivada analítica: 2 (la pendiente de una línea)")

print("\nFunción f(x) = sin(x)")
print(f"Derivada en x=0: {derivada_numerica(f3, 0):.4f}")
print(f"Derivada analítica: {np.cos(0):.4f} (fórmula: cos(x))")

# ============================================================================
# 3. GRADIENTE DESCENDENTE - EL CORAZÓN DEL APRENDIZAJE
# ============================================================================
print("\n\n3️⃣  GRADIENTE DESCENDENTE")
print("-" * 60)
print("""
💡 Concepto clave:
Imagina que estás en una montaña con niebla y quieres bajar.
No puedes ver el camino completo, pero puedes:
1. Sentir la pendiente bajo tus pies (calcular el gradiente)
2. Dar un paso hacia abajo (actualizar parámetros)
3. Repetir hasta llegar al valle (mínimo error)

En IA, la "altura" es el error del modelo, y queremos llegar al punto más bajo.
""")

def funcion_error(w):
    """
    Una función de error simple: (w - 5)²
    
    ¿Por qué el óptimo es 5?
    Porque esta fórmula mide la distancia al número 5.
    - Si w = 5: (5 - 5)² = 0  (Error mínimo, perfecto)
    - Si w = 4: (4 - 5)² = 1  (Hay error)
    - Si w = 10: (10 - 5)² = 25 (Mucho error)
    """
    return (w - 5)**2

def derivada_error(w):
    """
    Derivada del error respecto a w.
    Nos dice hacia dónde movernos para llegar a 5.
    """
    return 2 * (w - 5)

# Gradiente descendente
w_inicial = 0
learning_rate = 0.1
num_iteraciones = 160

# Guardamos el historial para visualizar
w_historia = [w_inicial]
error_historia = [funcion_error(w_inicial)]

w = w_inicial

print(f"{'Iter':<6} {'w':<10} {'Error':<10} {'Gradiente':<10}")
print("-" * 40)

for i in range(num_iteraciones):
    gradiente = derivada_error(w)
    w_nuevo = w - learning_rate * gradiente
    error = funcion_error(w_nuevo)
    
    w_historia.append(w_nuevo)
    error_historia.append(error)
    
    if i < 10 or i == num_iteraciones - 1:  # Mostrar primeras 10 y última
        print(f"{i+1:<6} {w_nuevo:<10.4f} {error:<10.4f} {gradiente:<10.4f}")
    
    w = w_nuevo

print(f"\n✅ Convergió a w = {w:.4f}")
print(f"💡 El valor óptimo es w = 5 (donde el error es cero)")

# ============================================================================
# 4. VISUALIZACIÓN DEL GRADIENTE DESCENDENTE
# ============================================================================
print("\n\n4️⃣  VISUALIZACIÓN")
print("-" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Función de error con ruta del gradiente
w_range = np.linspace(-2, 12, 100)
error_range = [funcion_error(w) for w in w_range]

ax1.plot(w_range, error_range, 'b-', linewidth=2, label='Error')
ax1.plot(w_historia, error_historia, 'ro-', markersize=8, linewidth=1, 
         alpha=0.7, label='Ruta del gradiente')
ax1.plot(w_historia[0], error_historia[0], 'go', markersize=15, 
         label='Inicio', zorder=5)
ax1.plot(w_historia[-1], error_historia[-1], 'r*', markersize=20, 
         label='Final', zorder=5)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('w (parámetro)')
ax1.set_ylabel('Error')
ax1.set_title('Gradiente Descendente en Acción')
ax1.legend()

# Subplot 2: Error vs iteración
ax2.plot(error_historia, 'b-', linewidth=2, marker='o', markersize=6)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Iteración')
ax2.set_ylabel('Error')
ax2.set_title('Convergencia del Error')
ax2.set_yscale('log')  # Escala logarítmica para ver mejor

plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/01-fundamentos-matematicos/gradiente_descendente.png', dpi=100)
print("✅ Gráfico guardado como 'gradiente_descendente.png'")

# ============================================================================
# 5. EJEMPLO PRÁCTICO: REGRESIÓN LINEAL
# ============================================================================
print("\n\n5️⃣  APLICACIÓN: REGRESIÓN LINEAL CON GRADIENTE DESCENDENTE")
print("-" * 60)

# Datos de ejemplo: y = 2x + 1 + ruido
np.random.seed(42)
X = np.linspace(0, 10, 50)
y_verdadero = 2 * X + 1
ruido = np.random.normal(0, 1, 50)
y = y_verdadero + ruido

# Queremos encontrar los mejores valores de m (pendiente) y b (intercepto)
# para la ecuación: y = m*x + b

def prediccion(X, m, b):
    return m * X + b

def error_mse(y_pred, y_real):
    """Error cuadrático medio"""
    return np.mean((y_pred - y_real)**2)

def gradientes(X, y, m, b):
    """
    Calcula los gradientes del error respecto a m y b
    No necesitas memorizar estas fórmulas, solo entender que:
    - Miden cuánto cambiar m y b para reducir el error
    """
    n = len(X)
    y_pred = prediccion(X, m, b)
    
    # Gradiente respecto a m
    dm = (-2/n) * np.sum(X * (y - y_pred))
    # Gradiente respecto a b
    db = (-2/n) * np.sum(y - y_pred)
    
    return dm, db

# Inicialización aleatoria
m, b = 0.0, 0.0
lr = 0.01
epochs = 500

m_historia = [m]
b_historia = [b]
error_historia_regresion = []

for epoch in range(epochs):
    # Predicción actual
    y_pred = prediccion(X, m, b)
    
    # Error
    error = error_mse(y_pred, y)
    error_historia_regresion.append(error)
    
    # Gradientes
    dm, db = gradientes(X, y, m, b)
    
    # Actualización
    m = m - lr * dm
    b = b - lr * db
    
    m_historia.append(m)
    b_historia.append(b)
    
    if epoch % 20 == 0:
        print(f"Época {epoch:3d}: m={m:.4f}, b={b:.4f}, Error={error:.4f}")

print(f"\n✅ Resultado final: y = {m:.4f}x + {b:.4f}")
print(f"💡 Valores reales: y = 2.0x + 1.0")

# Visualización final
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Datos y línea ajustada
axes[0].scatter(X, y, alpha=0.6, label='Datos')
axes[0].plot(X, y_verdadero, 'g--', linewidth=2, label='Línea real')
axes[0].plot(X, prediccion(X, m, b), 'r-', linewidth=2, label='Línea ajustada')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('Regresión Lineal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Convergencia del error
axes[1].plot(error_historia_regresion, 'b-', linewidth=2)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Error MSE')
axes[1].set_title('Reducción del Error')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

# Evolución de parámetros
axes[2].plot(m_historia, label='m (pendiente)', linewidth=2)
axes[2].plot(b_historia, label='b (intercepto)', linewidth=2)
axes[2].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='m real')
axes[2].axhline(y=1, color='g', linestyle='--', alpha=0.5, label='b real')
axes[2].set_xlabel('Época')
axes[2].set_ylabel('Valor del parámetro')
axes[2].set_title('Convergencia de parámetros')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/01-fundamentos-matematicos/regresion_lineal.png', dpi=100)
print("✅ Gráfico guardado como 'regresion_lineal.png'")

# ============================================================================
# 6. EJERCICIOS
# ============================================================================
print("\n\n📝 EJERCICIOS PARA PRACTICAR")
print("-" * 60)
print("""
1. Modifica el learning_rate en el ejemplo de gradiente descendente.
   ¿Qué pasa si es muy grande (0.5)? ¿Y si es muy pequeño (0.001)?

2. Implementa gradiente descendente para minimizar f(x) = x^4 - 2x^2 + 1
   Pista: usa derivada_numerica() para calcular el gradiente

3. En la regresión lineal, aumenta el número de epochs a 500.
   ¿Mejora el resultado?

4. Genera tus propios datos con una relación y = 3x - 5
   y entrena el modelo para encontrar estos parámetros.

5. Investiga qué pasa si usas diferentes valores iniciales para m y b.
   ¿Siempre converge al mismo resultado?
""")

print("\n" + "=" * 60)
print("¡Has completado la lección de Cálculo y Gradientes!")
print("Ahora entiendes el fundamento del entrenamiento de IA 🎉")
print("Siguiente: 03-probabilidad-basica.py")
print("=" * 60)

plt.show()
