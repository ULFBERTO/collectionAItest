---
language: es
tags:
- text-generation
- gpt
- tensorflow
- don-quijote
license: mit
---

# GPT OxideLLM_5M

Modelo de lenguaje tipo GPT entrenado desde cero con el texto de OxideLLM_5M de la Mancha.

## Configuración del modelo

- **Vocabulario**: 221 caracteres
- **Dimensión del modelo (d_model)**: 256
- **Número de cabezas de atención**: 4
- **Dimensión feedforward (dff)**: 512
- **Número de capas**: 4
- **Longitud máxima de secuencia**: 128

## Uso

```python
from huggingface_hub import snapshot_download
import tensorflow as tf
import json

# Descargar modelo
model_path = snapshot_download(repo_id="ULFBERTO/OxideLLM_5M")

# Cargar vocabulario
with open(f"{model_path}/vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
    
char2idx = vocab_data["char2idx"]
idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}

# Cargar modelo
model = tf.keras.models.load_model(f"{model_path}/saved_model")
```

## Entrenamiento

Entrenado con TensorFlow 2.x usando arquitectura Transformer (decoder-only).
