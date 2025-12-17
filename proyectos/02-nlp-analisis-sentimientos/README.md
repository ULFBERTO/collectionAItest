# 💬 Proyecto: Análisis de Sentimientos (NLP Básico)

## 📋 Objetivo

Crear un clasificador de sentimientos que determine si un texto es positivo, negativo o neutral usando técnicas de NLP y Machine Learning.

## 🎯 Lo que aprenderás

- ✅ Preprocesamiento de texto
- ✅ Tokenización y vectorización
- ✅ TF-IDF y Bag of Words
- ✅ Word embeddings básicos
- ✅ Clasificación de texto
- ✅ Evaluación de modelos NLP

## 🔧 Requisitos

```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud
```

## 📁 Archivos del Proyecto

- `01-analisis-sentimientos-basico.py` - Clasificador con scikit-learn
- `02-analisis-sentimientos-deep.py` - Red neuronal con embeddings
- `README.md` - Esta guía

## 🚀 Ejecución Rápida

```bash
# Ejemplo básico con ML tradicional
python 01-analisis-sentimientos-basico.py

# Ejemplo avanzado con Deep Learning
python 02-analisis-sentimientos-deep.py
```

## 📊 Dataset

Usaremos reseñas de productos/películas:
- **Positivas**: "¡Excelente producto! Lo recomiendo"
- **Negativas**: "Muy malo, no funciona"
- **Neutrales**: "Es un producto normal"

## 🎓 Conceptos Clave

### 1. Tokenización
Dividir texto en palabras/tokens:
```
"Me encanta este producto" → ["Me", "encanta", "este", "producto"]
```

### 2. Vectorización
Convertir texto en números:
- **Bag of Words**: Frecuencia de palabras
- **TF-IDF**: Importancia relativa de palabras

### 3. Embeddings
Representar palabras como vectores densos que capturan significado semántico.

## 📈 Resultados Esperados

- **Modelo básico (TF-IDF + Logistic Regression)**: ~75-85% precisión
- **Modelo deep (Embeddings + LSTM)**: ~85-92% precisión

---

**Tiempo estimado:** 2-3 horas
