# 📐 Fundamentos Matemáticos para IA

No necesitas recordar todas las ecuaciones de memoria. Lo importante es **entender los conceptos** y saber aplicarlos con código.

## 🎯 Objetivos

Al finalizar esta sección, comprenderás:
- ✅ Operaciones con vectores y matrices
- ✅ Por qué son importantes en IA
- ✅ Gradientes y derivadas (sin ecuaciones complicadas)
- ✅ Cómo se usan en el entrenamiento de modelos

## 📚 Contenido

### 1. Álgebra Lineal Simplificada

#### 🔢 Vectores
Un vector es simplemente una lista de números:

```python
# En Python con NumPy
import numpy as np

vector = np.array([1, 2, 3, 4])
print(f"Vector: {vector}")
```

**¿Por qué son importantes?**
- Cada imagen es un vector de píxeles
- Cada palabra se puede representar como un vector
- Los modelos de IA procesan vectores

#### 📊 Matrices
Una matriz es una tabla de números:

```python
matriz = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(f"Matriz:\n{matriz}")
```

**¿Por qué son importantes?**
- Los pesos de una red neuronal son matrices
- Las transformaciones de datos usan matrices

#### ⚡ Operaciones Básicas

```python
# Suma de vectores
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
suma = v1 + v2  # [5, 7, 9]

# Producto punto (dot product)
producto = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Multiplicación de matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
```

### 2. Cálculo para IA (Sin Miedo)

#### 📉 Derivadas - ¿Qué son?
La derivada te dice **cuánto cambia algo**.

**Ejemplo intuitivo:**
- Si tienes un error de 10 y después de 8
- La derivada te dice que **estás mejorando** (-2)

```python
# Ejemplo práctico: función cuadrática
def funcion(x):
    return x**2

# La derivada de x^2 es 2*x
def derivada(x):
    return 2*x

# Si estás en x=3, la derivada es 6
# Esto significa: si aumentas x un poco, y aumenta 6 veces más
x = 3
print(f"En x={x}, la función vale {funcion(x)}")
print(f"La derivada (tasa de cambio) es {derivada(x)}")
```

#### 🎯 Gradiente Descendente
Es la técnica para **minimizar el error** de un modelo.

**Analogía:** Imagina que estás en una montaña con niebla y quieres bajar:
1. Miras a tu alrededor (calculas el gradiente)
2. Te mueves hacia donde baja más (sigues el gradiente)
3. Repites hasta llegar abajo (mínimo error)

```python
# Ejemplo simple de gradiente descendente
def error(w):
    # Error cuadrático simple
    return (w - 5)**2

def derivada_error(w):
    # Derivada del error respecto a w
    return 2 * (w - 5)

# Proceso de optimización
w = 0  # Valor inicial
learning_rate = 0.1  # Tamaño del paso

for i in range(10):
    gradiente = derivada_error(w)
    w = w - learning_rate * gradiente  # Actualización
    print(f"Iteración {i+1}: w={w:.2f}, error={error(w):.2f}")

# w converge a 5 (donde el error es mínimo)
```

### 3. Probabilidad Básica

#### 🎲 Conceptos Clave

```python
import numpy as np

# Simulación de lanzamientos de moneda
lanzamientos = np.random.choice(['cara', 'cruz'], size=1000)
probabilidad_cara = np.sum(lanzamientos == 'cara') / 1000
print(f"Probabilidad de cara: {probabilidad_cara:.2f}")

# Distribución normal (muy usada en IA)
datos = np.random.normal(loc=0, scale=1, size=1000)
# loc = media, scale = desviación estándar
```

## 🏋️ Ejercicios Prácticos

### Ejercicio 1: Operaciones con Vectores
```python
# TODO: Implementa estas funciones
def magnitud_vector(v):
    """Calcula la magnitud (longitud) de un vector"""
    pass

def similitud_coseno(v1, v2):
    """Calcula qué tan similares son dos vectores (0 a 1)"""
    pass
```

### Ejercicio 2: Gradiente Descendente
```python
# TODO: Implementa gradiente descendente para una función lineal
def gradiente_descendente_lineal(X, y, epochs=100, lr=0.01):
    """
    Encuentra la mejor línea que se ajusta a los datos
    X: datos de entrada
    y: valores objetivo
    epochs: número de iteraciones
    lr: learning rate
    """
    pass
```

## 📖 Recursos Adicionales

- **Cheat Sheet**: Ver [`algebra-lineal-cheatsheet.md`](./algebra-lineal-cheatsheet.md)
- **Visualizaciones**: Ver [`visualizaciones.ipynb`](./visualizaciones.ipynb)
- **Soluciones**: Ver [`soluciones/`](./soluciones/)

## ✅ Autoevaluación

Antes de continuar a la siguiente fase, asegúrate de poder:

- [ ] Crear y manipular vectores y matrices en NumPy
- [ ] Entender qué es un producto punto y por qué es útil
- [ ] Explicar qué es una derivada con tus propias palabras
- [ ] Implementar gradiente descendente simple
- [ ] Generar números aleatorios con distribuciones

---

**Siguiente:** [Fase 2 - Python para IA](../02-python-para-ia/README.md)
