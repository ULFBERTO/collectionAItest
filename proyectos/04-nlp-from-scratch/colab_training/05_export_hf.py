# ============================================================
# SECCIÓN 5: EXPORTAR A HUGGINGFACE
# ============================================================

import os
import json
import shutil
from huggingface_hub import HfApi, login, create_repo

def export_to_huggingface(
    model,
    tokenizer_path,
    output_path,
    repo_id="ULFBERTO/OxideLLM_5M",
    hf_token=None
):
    """
    Exporta el modelo a HuggingFace Hub.
    
    Args:
        model: Modelo entrenado
        tokenizer_path: Ruta al tokenizer SentencePiece
        output_path: Ruta donde guardar archivos temporales
        repo_id: ID del repositorio en HuggingFace
        hf_token: Token de HuggingFace (o usar login interactivo)
    """
    
    # Login a HuggingFace
    if hf_token:
        login(token=hf_token)
    else:
        login()  # Login interactivo
    
    api = HfApi()
    
    # Crear directorio de exportación
    export_dir = os.path.join(output_path, "hf_export")
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"📦 Preparando exportación a {repo_id}...")
    
    # 1. Guardar modelo en formato SavedModel
    model_dir = os.path.join(export_dir, "saved_model")
    model.save(model_dir)
    print("  ✓ SavedModel guardado")
    
    # 2. Guardar pesos en formato H5
    weights_path = os.path.join(export_dir, "model_weights.h5")
    model.save_weights(weights_path)
    print("  ✓ Pesos H5 guardados")
    
    # 3. Copiar tokenizer
    tokenizer_model = os.path.join(tokenizer_path, "oxide_bpe.model")
    tokenizer_vocab = os.path.join(tokenizer_path, "oxide_bpe.vocab")
    
    if os.path.exists(tokenizer_model):
        shutil.copy(tokenizer_model, os.path.join(export_dir, "tokenizer.model"))
        print("  ✓ Tokenizer copiado")
    
    if os.path.exists(tokenizer_vocab):
        shutil.copy(tokenizer_vocab, os.path.join(export_dir, "tokenizer.vocab"))

    # 4. Crear config.json
    config = model.get_config()
    config["model_type"] = "oxide_llm"
    config["architecture"] = "OxideLLM"
    
    config_path = os.path.join(export_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print("  ✓ Config guardado")
    
    # 5. Crear README
    readme_content = f"""---
license: mit
language:
- es
tags:
- text-generation
- tensorflow
- spanish
- gpt
library_name: tensorflow
---

# OxideLLM

Modelo de lenguaje en español entrenado desde cero.

## Arquitectura

- **Tipo**: Transformer decoder-only (GPT-style)
- **Parámetros**: ~{model.count_params()/1e6:.1f}M
- **Vocabulario**: {config['vocab_size']} tokens (BPE)
- **Contexto**: {config['max_len']} tokens
- **Capas**: {config['num_layers']}
- **Dimensión**: {config['d_model']}
- **Cabezas de atención**: {config['num_heads']}

## Uso

```python
import tensorflow as tf
import sentencepiece as spm

# Cargar tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# Cargar modelo
model = tf.keras.models.load_model("saved_model")

# Generar texto
prompt = "En un lugar de la Mancha"
tokens = sp.encode(prompt)
# ... (ver código de generación)
```

## Entrenamiento

Entrenado con literatura clásica española (Cervantes, Galdós, Clarín, etc.)

## Licencia

MIT
"""
    
    readme_path = os.path.join(export_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("  ✓ README creado")
    
    # 6. Crear/actualizar repositorio
    try:
        create_repo(repo_id, exist_ok=True)
        print(f"  ✓ Repositorio {repo_id} listo")
    except Exception as e:
        print(f"  ⚠ Repo ya existe o error: {e}")
    
    # 7. Subir archivos
    print("\n📤 Subiendo a HuggingFace...")
    
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update OxideLLM model"
    )
    
    print(f"\n✅ Modelo subido exitosamente a: https://huggingface.co/{repo_id}")
    return export_dir


# Uso:
# export_to_huggingface(
#     model=model,
#     tokenizer_path=TOKENIZER_PATH,
#     output_path=OUTPUT_PATH,
#     repo_id="ULFBERTO/OxideLLM_5M",
#     hf_token="hf_xxxxx"  # O dejar None para login interactivo
# )
