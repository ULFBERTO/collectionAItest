# OxideLLM Training en Google Colab

Este directorio contiene los scripts para entrenar OxideLLM en Google Colab.

## Archivos

1. `01_setup.py` - Configuración inicial y dependencias
2. `02_data_loader.py` - Carga y preprocesamiento de datos con BPE
3. `03_model.py` - Arquitectura mejorada del modelo
4. `04_train.py` - Loop de entrenamiento
5. `05_export_hf.py` - Exportar a HuggingFace
6. `06_convert_tfjs.py` - Convertir a TFJS

## Uso en Colab

1. Sube la carpeta `libros_espanol` a tu Google Drive
2. Ejecuta cada sección en orden en un notebook de Colab
3. Asegúrate de tener tu token de HuggingFace configurado

## Mejoras sobre el modelo original

- Tokenización BPE (8000 tokens) en lugar de caracteres
- Contexto de 512 tokens (vs 100 caracteres)
- Modelo más grande: ~50M parámetros
- Arquitectura GPT-2 style con Pre-LN
