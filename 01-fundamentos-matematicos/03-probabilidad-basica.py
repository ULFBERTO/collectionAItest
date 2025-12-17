"""
Probabilidad Básica para IA
===========================

La incertidumbre es fundamental en IA. Los modelos no dicen "es un gato",
dicen "hay un 95% de probabilidad de que sea un gato".
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Para gráficos más bonitos de distribuciones

print("=" * 60)
print("PROBABILIDAD BÁSICA - LA BASE DE LA INCERTIDUMBRE")
print("=" * 60)

# ============================================================================
# 1. PROBABILIDAD SIMPLE
# ============================================================================
print("\n1️⃣  PROBABILIDAD SIMPLE (MONEDAS Y DADOS)")
print("-" * 60)

# Simulación de lanzamiento de moneda
# 0 = Cruz, 1 = Cara
n_lanzamientos = 9999999
monedas = np.random.randint(0, 2, n_lanzamientos)

print(f"Lanzamos una moneda {n_lanzamientos} veces:")
print(f"Resultados: {monedas}")
print(f"Caras (1): {np.sum(monedas)}")
print(f"Probabilidad observada de cara: {np.mean(monedas):.2f}")
print("Probabilidad teórica: 0.50")

print("\n💡 Ley de los Grandes Números:")
print("Si lanzamos pocas veces, el resultado varía mucho.")
print("Si lanzamos muchas veces, se acerca a la probabilidad real.")

# Lanzar 1000 veces
muchas_monedas = np.random.randint(0, 2, 1000)
print(f"\nLanzando 1000 veces...")
print(f"Probabilidad observada: {np.mean(muchas_monedas):.4f}")

# ============================================================================
# 2. DISTRIBUCIONES DE PROBABILIDAD (LA CURVA DE CAMPANA)
# ============================================================================
print("\n\n2️⃣  DISTRIBUCIÓN NORMAL (GAUSSIANA)")
print("-" * 60)
print("""
La mayoría de las cosas en la naturaleza siguen una 'Curva de Campana':
- Altura de las personas
- Notas de exámenes
- Errores en mediciones

En IA, asumimos que los datos siguen esta forma muchas veces.
""")

# Generar datos con distribución normal
# Media (mu) = 0, Desviación estándar (sigma) = 1
datos_normales = np.random.normal(loc=0, scale=1, size=10000)

print(f"Media de los datos: {np.mean(datos_normales):.4f} (debería ser ~0)")
print(f"Desviación estándar: {np.std(datos_normales):.4f} (debería ser ~1)")

# Visualización
plt.figure(figsize=(10, 6))
plt.hist(datos_normales, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

# Dibujar la curva teórica encima
x = np.linspace(-4, 4, 100)
p = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
plt.plot(x, p, 'r-', linewidth=2, label='Curva Teórica')

plt.title('Distribución Normal Estándar')
plt.xlabel('Valor')
plt.ylabel('Probabilidad (Densidad)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('d:/EVIROMENT/PracticaIA/01-fundamentos-matematicos/distribucion_normal.png')
print("✅ Gráfico guardado como 'distribucion_normal.png'")

# ============================================================================
# 3. PROBABILIDAD CONDICIONAL (BAYES SIMPLIFICADO)
# ============================================================================
print("\n\n3️⃣  PROBABILIDAD CONDICIONAL")
print("-" * 60)
print("""
P(A | B) -> Probabilidad de A DADO QUE pasó B.

Ejemplo: Detectar Spam
- P(Spam): Probabilidad de que un correo sea spam en general.
- P(Palabra="Oferta" | Spam): Si es spam, ¿qué tan probable es que diga "Oferta"?
""")

# Ejemplo simple:
# Tenemos 100 correos.
# 40 son Spam.
# De los 40 de Spam, 30 tienen la palabra "Oferta".
# De los 60 que NO son Spam, solo 5 tienen la palabra "Oferta".

total_correos = 100
spam = 40
no_spam = 60

spam_con_oferta = 30
no_spam_con_oferta = 5

# Pregunta: Si veo la palabra "Oferta", ¿cuál es la probabilidad de que sea Spam?
# P(Spam | Oferta) = P(Spam y Oferta) / P(Oferta)

total_con_oferta = spam_con_oferta + no_spam_con_oferta
prob_spam_dado_oferta = spam_con_oferta / total_con_oferta

print(f"Total correos con 'Oferta': {total_con_oferta}")
print(f"De esos, son Spam: {spam_con_oferta}")
print(f"Probabilidad de ser Spam si dice 'Oferta': {prob_spam_dado_oferta:.2%}")
print("💡 ¡Por eso los filtros de spam funcionan buscando palabras clave!")

# ============================================================================
# 4. EJERCICIOS
# ============================================================================
print("\n\n📝 EJERCICIOS PARA PRACTICAR")
print("-" * 60)
print("""
1. Modifica la desviación estándar (scale) en la distribución normal.
   ¿Qué pasa con la gráfica si pones scale=0.5? ¿Y scale=2?

2. Simula el lanzamiento de dos dados (suma de dos números del 1 al 6).
   Haz un histograma con 1000 lanzamientos. ¿Qué forma tiene?
   (Pista: No es plana, es triangular/campana).

3. Calcula la probabilidad de obtener un número mayor a 0.5 en la distribución normal.
   (Pista: cuenta cuántos datos son > 0.5 y divide por el total).
""")

print("\n" + "=" * 60)
print("¡Has completado la lección de Probabilidad!")
print("Siguiente: 04-estadistica-descriptiva.py")
print("=" * 60)

# plt.show() # Descomentar si quieres ver la ventana emergente
