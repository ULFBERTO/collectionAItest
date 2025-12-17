"""
Estadística Descriptiva para IA
===============================

Antes de entrenar cualquier modelo, DEBES entender tus datos.
La estadística descriptiva te ayuda a resumir y visualizar qué está pasando.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Usaremos pandas, es el estándar para datos tabulares

print("=" * 60)
print("ESTADÍSTICA DESCRIPTIVA - ENTENDIENDO TUS DATOS")
print("=" * 60)

# ============================================================================
# 1. MEDIDAS DE TENDENCIA CENTRAL (¿DÓNDE ESTÁ EL CENTRO?)
# ============================================================================
print("\n1️⃣  MEDIA, MEDIANA Y MODA")
print("-" * 60)

# Imaginemos salarios en una empresa pequeña (en miles $)
salarios = np.array([30, 35, 32, 31, 33, 30, 34, 30, 300]) # ¡Ojo con el 300! (El jefe)

print(f"Datos de salarios: {salarios}")

media = np.mean(salarios)
mediana = np.median(salarios)

# La moda es más fácil con pandas o scipy, aquí lo hacemos manual para entender
valores, conteos = np.unique(salarios, return_counts=True)
moda = valores[np.argmax(conteos)]

print(f"\nMedia (Promedio): {media:.2f}")
print(f"Mediana (Valor central): {mediana:.2f}")
print(f"Moda (Más frecuente): {moda}")

print("\n💡 LECCIÓN IMPORTANTE:")
print("La MEDIA es sensible a valores extremos (outliers) como el 300.")
print("La MEDIANA es robusta. Si hay valores locos, usa la mediana.")

# ============================================================================
# 2. MEDIDAS DE DISPERSIÓN (¿QUÉ TAN SEPARADOS ESTÁN?)
# ============================================================================
print("\n\n2️⃣  VARIANZA Y DESVIACIÓN ESTÁNDAR")
print("-" * 60)

# Dos grupos con la misma media pero diferente dispersión
grupo_A = np.array([48, 49, 50, 51, 52]) # Muy parecidos
grupo_B = np.array([10, 30, 50, 70, 90]) # Muy diferentes

print(f"Grupo A: {grupo_A} (Media: {np.mean(grupo_A)})")
print(f"Grupo B: {grupo_B} (Media: {np.mean(grupo_B)})")

var_A = np.var(grupo_A)
std_A = np.std(grupo_A)

var_B = np.var(grupo_B)
std_B = np.std(grupo_B)

print(f"\nDesviación Estándar A: {std_A:.2f}")
print(f"Desviación Estándar B: {std_B:.2f}")
print("💡 La desviación estándar te dice cuánto se alejan los datos del promedio.")

# ============================================================================
# 3. CUARTILES Y BOXPLOTS (DETECTANDO OUTLIERS)
# ============================================================================
print("\n\n3️⃣  CUARTILES Y PERCENTILES")
print("-" * 60)

datos = np.random.normal(100, 20, 1000) # Media 100, Desviación 20

# Percentiles clave
q1 = np.percentile(datos, 25) # 25% de los datos están por debajo
q2 = np.percentile(datos, 50) # 50% (Mediana)
q3 = np.percentile(datos, 75) # 75%

print(f"Q1 (25%): {q1:.2f}")
print(f"Q2 (Mediana): {q2:.2f}")
print(f"Q3 (75%): {q3:.2f}")
print(f"Rango Intercuartílico (IQR): {q3 - q1:.2f}")

# Visualización: Boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(datos, vert=False)
plt.title('Boxplot (Diagrama de Caja)')
plt.xlabel('Valor')
plt.yticks([])
plt.grid(True, alpha=0.3)

# Añadir texto explicativo al gráfico
plt.text(q1, 1.1, 'Q1', ha='center')
plt.text(q2, 1.1, 'Mediana', ha='center')
plt.text(q3, 1.1, 'Q3', ha='center')

plt.savefig('d:/EVIROMENT/PracticaIA/01-fundamentos-matematicos/boxplot.png')
print("✅ Gráfico guardado como 'boxplot.png'")

# ============================================================================
# 4. CORRELACIÓN (¿SE MUEVEN JUNTOS?)
# ============================================================================
print("\n\n4️⃣  CORRELACIÓN")
print("-" * 60)

# Generamos datos correlacionados
x = np.linspace(0, 10, 50)
y_pos = 2 * x + np.random.normal(0, 2, 50) # Correlación positiva fuerte
y_neg = -2 * x + np.random.normal(0, 2, 50) # Correlación negativa fuerte
y_no = np.random.normal(0, 10, 50) # Sin correlación

corr_pos = np.corrcoef(x, y_pos)[0, 1]
corr_neg = np.corrcoef(x, y_neg)[0, 1]
corr_no = np.corrcoef(x, y_no)[0, 1]

print(f"Correlación Positiva: {corr_pos:.2f} (Cerca de 1)")
print(f"Correlación Negativa: {corr_neg:.2f} (Cerca de -1)")
print(f"Sin Correlación: {corr_no:.2f} (Cerca de 0)")

# Visualización
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(x, y_pos, color='green')
ax1.set_title(f'Positiva (r={corr_pos:.2f})')

ax2.scatter(x, y_neg, color='red')
ax2.set_title(f'Negativa (r={corr_neg:.2f})')

ax3.scatter(x, y_no, color='gray')
ax3.set_title(f'Ninguna (r={corr_no:.2f})')

plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/01-fundamentos-matematicos/correlacion.png')
print("✅ Gráfico guardado como 'correlacion.png'")

# ============================================================================
# 5. EJERCICIOS
# ============================================================================
print("\n\n📝 EJERCICIOS PARA PRACTICAR")
print("-" * 60)
print("""
1. Crea un array con tus gastos mensuales estimados.
   Calcula la media y la desviación estándar.

2. Genera dos variables aleatorias que NO estén correlacionadas.
   Calcula su coeficiente de correlación para comprobarlo.

3. Investiga qué es la "Matriz de Correlación" en pandas (df.corr()).
   Es fundamental para seleccionar características (features) en Machine Learning.
""")

print("\n" + "=" * 60)
print("¡Has completado la lección de Estadística Descriptiva!")
print("Con esto terminamos el Módulo 1: Fundamentos Matemáticos 🎓")
print("=" * 60)
