"""
Análisis de Sentimientos - NLP Básico
======================================

Clasificador de sentimientos usando técnicas tradicionales de ML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ANÁLISIS DE SENTIMIENTOS - NLP BÁSICO")
print("=" * 70)

# ============================================================================
# 1. DATASET DE EJEMPLO
# ============================================================================
print("\n1️⃣  CREANDO DATASET DE RESEÑAS")
print("-" * 70)

# Crear dataset de ejemplo (en producción usarías datos reales)
reseñas_positivas = [
    "Excelente producto, lo recomiendo totalmente",
    "Me encantó, superó mis expectativas",
    "Muy buena calidad, estoy muy satisfecho",
    "Increíble, es justo lo que necesitaba",
    "Perfecto, llegó rápido y funciona de maravilla",
    "Fantástico, vale cada peso que pagué",
    "Maravilloso, mi familia está encantada",
    "Estupendo, lo volvería a comprar sin dudarlo",
    "Genial, es de muy buena calidad",
    "Extraordinario, supera a productos más caros"
] * 20  # Repetir para tener más datos

reseñas_negativas = [
    "Muy malo, no sirve para nada",
    "Terrible calidad, una decepción total",
    "No lo recomiendo, es muy deficiente",
    "Horrible, llegó roto y no funciona",
    "Pésimo, perdí mi dinero",
    "Muy decepcionante, esperaba mucho más",
    "No vale la pena, busquen otra opción",
    "Defectuoso, tuve que devolverlo",
    "Mala experiencia, no lo compren",
    "Insatisfecho completamente, no cumple lo prometido"
] * 20

reseñas_neutrales = [
    "Es un producto normal, nada especial",
    "Cumple su función básica",
    "Ni bueno ni malo, es estándar",
    "Para el precio, está bien",
    "Es lo que se esperaba",
    "Funciona correctamente",
    "Un producto más del mercado",
    "Aceptable para uso casual",
    "Cumple con lo mínimo",
    "Sin grandes sorpresas"
] * 20

# Combinar y crear DataFrame
textos = reseñas_positivas + reseñas_negativas + reseñas_neutrales
etiquetas = ['positivo'] * len(reseñas_positivas) + \
            ['negativo'] * len(reseñas_negativas) + \
            ['neutral'] * len(reseñas_neutrales)

df = pd.DataFrame({
    'texto': textos,
    'sentimiento': etiquetas
})

# Mezclar datos
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total de reseñas: {len(df)}")
print(f"\nDistribución de sentimientos:")
print(df['sentimiento'].value_counts())

# ============================================================================
# 2. PREPROCESAMIENTO DE TEXTO
# ============================================================================
print("\n\n2️⃣  PREPROCESAMIENTO DE TEXTO")
print("-" * 70)

def limpiar_texto(texto):
    """
    Limpia y normaliza el texto.
    """
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Remover caracteres especiales y números
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    
    # Remover espacios extras
    texto = ' '.join(texto.split())
    
    return texto

# Aplicar limpieza
df['texto_limpio'] = df['texto'].apply(limpiar_texto)

print("Ejemplo de limpieza:")
print(f"Original:  {df['texto'].iloc[0]}")
print(f"Limpio:    {df['texto_limpio'].iloc[0]}")

# ============================================================================
# 3. DIVISIÓN DE DATOS
# ============================================================================
print("\n\n3️⃣  DIVISIÓN DE DATOS")
print("-" * 70)

X = df['texto_limpio']
y = df['sentimiento']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Conjunto de entrenamiento: {len(X_train)}")
print(f"Conjunto de prueba: {len(X_test)}")

# ============================================================================
# 4. MODELO 1: TF-IDF + LOGISTIC REGRESSION
# ============================================================================
print("\n\n4️⃣  MODELO 1: TF-IDF + REGRESIÓN LOGÍSTICA")
print("-" * 70)

# Crear pipeline
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Entrenar
print("Entrenando...")
pipeline_lr.fit(X_train, y_train)

# Predecir
y_pred_lr = pipeline_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"\n✅ Precisión: {accuracy_lr*100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_lr))

# ============================================================================
# 5. MODELO 2: BAG OF WORDS + NAIVE BAYES
# ============================================================================
print("\n\n5️⃣  MODELO 2: BAG OF WORDS + NAIVE BAYES")
print("-" * 70)

pipeline_nb = Pipeline([
    ('vectorizer', CountVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

print("Entrenando...")
pipeline_nb.fit(X_train, y_train)

y_pred_nb = pipeline_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"\n✅ Precisión: {accuracy_nb*100:.2f}%")

# ============================================================================
# 6. MODELO 3: TF-IDF + SVM
# ============================================================================
print("\n\n6️⃣  MODELO 3: TF-IDF + SVM")
print("-" * 70)

pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('classifier', LinearSVC(random_state=42))
])

print("Entrenando...")
pipeline_svm.fit(X_train, y_train)

y_pred_svm = pipeline_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"\n✅ Precisión: {accuracy_svm*100:.2f}%")

# ============================================================================
# 7. COMPARACIÓN DE MODELOS
# ============================================================================
print("\n\n7️⃣  COMPARACIÓN DE MODELOS")
print("-" * 70)

resultados = pd.DataFrame({
    'Modelo': ['Logistic Regression', 'Naive Bayes', 'SVM'],
    'Precisión': [accuracy_lr, accuracy_nb, accuracy_svm]
})

print(resultados.to_string(index=False))

# Visualización
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
axes[0].barh(resultados['Modelo'], resultados['Precisión']*100, 
             color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black', linewidth=2)
axes[0].set_xlabel('Precisión (%)', fontsize=11)
axes[0].set_title('Comparación de Modelos', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, 100)
axes[0].grid(True, alpha=0.3, axis='x')

for i, (modelo, prec) in enumerate(zip(resultados['Modelo'], resultados['Precisión'])):
    axes[0].text(prec*100 + 1, i, f'{prec*100:.1f}%', 
                 va='center', fontsize=10, fontweight='bold')

# Matriz de confusión del mejor modelo
mejor_modelo_idx = resultados['Precisión'].idxmax()
mejor_modelo_nombre = resultados.loc[mejor_modelo_idx, 'Modelo']

if mejor_modelo_idx == 0:
    y_pred_mejor = y_pred_lr
elif mejor_modelo_idx == 1:
    y_pred_mejor = y_pred_nb
else:
    y_pred_mejor = y_pred_svm

cm = confusion_matrix(y_test, y_pred_mejor)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=pipeline_lr.classes_,
            yticklabels=pipeline_lr.classes_,
            ax=axes[1])
axes[1].set_title(f'Matriz de Confusión - {mejor_modelo_nombre}', 
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Real', fontsize=11)
axes[1].set_xlabel('Predicción', fontsize=11)

plt.tight_layout()
plt.savefig('d:/EVIROMENT/PracticaIA/proyectos/02-nlp-analisis-sentimientos/01-comparacion-modelos.png', 
            dpi=100, bbox_inches='tight')
print("\n✅ Guardado: 01-comparacion-modelos.png")

# ============================================================================
# 8. ANÁLISIS DE PALABRAS IMPORTANTES
# ============================================================================
print("\n\n8️⃣  PALABRAS MÁS IMPORTANTES")
print("-" * 70)

# Obtener vocabulario y coeficientes
vectorizer = pipeline_lr.named_steps['tfidf']
classifier = pipeline_lr.named_steps['classifier']

feature_names = vectorizer.get_feature_names_out()

# Para clasificación multiclase, obtenemos coeficientes por clase
for i, clase in enumerate(classifier.classes_):
    coef = classifier.coef_[i]
    top_indices = np.argsort(coef)[-10:][::-1]
    
    print(f"\nPalabras más importantes para '{clase}':")
    for idx in top_indices:
        print(f"  • {feature_names[idx]}: {coef[idx]:.3f}")

# ============================================================================
# 9. PREDICCIONES DE EJEMPLO
# ============================================================================
print("\n\n9️⃣  PREDICCIONES DE EJEMPLO")
print("-" * 70)

nuevos_textos = [
    "Este producto es increíble, me encantó todo",
    "Horrible, el peor producto que he comprado",
    "Es un producto normal, nada del otro mundo",
    "Excelente calidad, muy recomendado",
    "No funciona bien, estoy decepcionado"
]

print("\nProbando el mejor modelo:\n")
for texto in nuevos_textos:
    prediccion = pipeline_lr.predict([texto])[0]
    probabilidades = pipeline_lr.predict_proba([texto])[0]
    
    print(f"Texto: \"{texto}\"")
    print(f"→ Sentimiento: {prediccion.upper()}")
    print(f"  Probabilidades: ", end="")
    for clase, prob in zip(pipeline_lr.classes_, probabilidades):
        print(f"{clase}={prob:.2f} ", end="")
    print("\n")

# ============================================================================
# 10. RESUMEN
# ============================================================================
print("\n" + "=" * 70)
print("🎉 ¡ANÁLISIS COMPLETADO!")
print("=" * 70)

print("\n📚 LO QUE APRENDISTE:")
print("  ✅ Preprocesamiento de texto")
print("  ✅ Vectorización con TF-IDF y Bag of Words")
print("  ✅ Entrenamiento de clasificadores de texto")
print("  ✅ Evaluación de modelos NLP")
print("  ✅ Interpretación de características importantes")

print("\n🔍 CONCEPTOS CLAVE:")
print("  • TF-IDF: Mide importancia de palabras en documentos")
print("  • N-grams: Combinaciones de palabras (ej: 'muy bueno')")
print("  • Pipeline: Encadena preprocesamiento y modelo")
print("  • Logistic Regression: Excelente para clasificación de texto")

print("\n🚀 PRÓXIMOS PASOS:")
print("  1. Probar con un dataset real más grande")
print("  2. Implementar embeddings (Word2Vec, GloVe)")
print("  3. Usar redes neuronales (LSTMs)")
print("  4. Agregar more preprocessing (stemming, lemmatization)")
print("  5. Probar con transformers (BERT)")

print("\n" + "=" * 70)

plt.show()
