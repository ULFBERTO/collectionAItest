# ============================================================
# SECCIÓN 6: CONVERTIR A TFJS Y SUBIR
# ============================================================

import os
import subprocess
import shutil
from huggingface_hub import HfApi, login, create_repo

def convert_to_tfjs(saved_model_path, output_path):
    """
    Convierte SavedModel a TensorFlow.js.
    
    Args:
        saved_model_path: Ruta al SavedModel
        output_path: Ruta donde guardar el modelo TFJS
    """
    
    tfjs_dir = os.path.join(output_path, "tfjs_model")
    os.makedirs(tfjs_dir, exist_ok=True)
    
    print("🔄 Convirtiendo a TensorFlow.js...")
    
    # Usar tensorflowjs_converter
    cmd = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        saved_model_path,
        tfjs_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error en conversión: {result.stderr}")
        # Intentar método alternativo
        print("🔄 Intentando conversión alternativa...")
        cmd_alt = [
            "tensorflowjs_converter",
            "--input_format=keras_saved_model",
            saved_model_path,
            tfjs_dir
        ]
        result = subprocess.run(cmd_alt, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Conversión fallida: {result.stderr}")
    
    print(f"✅ Modelo TFJS guardado en: {tfjs_dir}")
    
    # Listar archivos generados
    files = os.listdir(tfjs_dir)
    print(f"📁 Archivos generados: {files}")
    
    return tfjs_dir


def upload_tfjs_to_huggingface(
    tfjs_dir,
    tokenizer_path,
    model_config,
    repo_id="ULFBERTO/OxideLLM_5M-tfjs",
    hf_token=None
):
    """
    Sube el modelo TFJS a HuggingFace.
    
    Args:
        tfjs_dir: Directorio con el modelo TFJS
        tokenizer_path: Ruta al tokenizer
        model_config: Configuración del modelo
        repo_id: ID del repositorio TFJS
        hf_token: Token de HuggingFace
    """
    
    if hf_token:
        login(token=hf_token)
    
    api = HfApi()
    
    # Crear directorio de exportación
    export_dir = os.path.join(os.path.dirname(tfjs_dir), "tfjs_export")
    os.makedirs(export_dir, exist_ok=True)
    
    # Copiar archivos TFJS
    for f in os.listdir(tfjs_dir):
        shutil.copy(os.path.join(tfjs_dir, f), export_dir)
    
    # Copiar tokenizer
    tokenizer_model = os.path.join(tokenizer_path, "oxide_bpe.model")
    tokenizer_vocab = os.path.join(tokenizer_path, "oxide_bpe.vocab")
    
    if os.path.exists(tokenizer_model):
        shutil.copy(tokenizer_model, os.path.join(export_dir, "tokenizer.model"))
    if os.path.exists(tokenizer_vocab):
        shutil.copy(tokenizer_vocab, os.path.join(export_dir, "tokenizer.vocab"))
    
    # Crear vocab.json para uso en JS
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_model)
    
    vocab = {}
    for i in range(sp.get_piece_size()):
        vocab[sp.id_to_piece(i)] = i
    
    import json
    vocab_path = os.path.join(export_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Guardar config
    config_path = os.path.join(export_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Crear README para TFJS
    readme = f"""---
license: mit
language:
- es
tags:
- text-generation
- tfjs
- tensorflow-js
- spanish
- browser
library_name: tensorflowjs
---

# OxideLLM - TensorFlow.js

Versión TFJS del modelo OxideLLM para ejecutar en navegador.

## Uso en JavaScript

```javascript
import * as tf from '@tensorflow/tfjs';

// Cargar modelo
const model = await tf.loadGraphModel('model.json');

// Cargar vocabulario
const vocab = await fetch('vocab.json').then(r => r.json());

// Tokenizar (simplificado - usar librería BPE real)
function tokenize(text) {{
  // Implementar tokenización BPE
}}

// Generar
async function generate(prompt, maxTokens = 50) {{
  let tokens = tokenize(prompt);
  
  for (let i = 0; i < maxTokens; i++) {{
    const input = tf.tensor2d([tokens.slice(-512)], [1, tokens.length]);
    const logits = model.predict(input);
    const lastLogits = logits.slice([0, tokens.length - 1, 0], [1, 1, -1]);
    const nextToken = tf.multinomial(lastLogits.squeeze(), 1).dataSync()[0];
    tokens.push(nextToken);
  }}
  
  return decode(tokens);
}}
```

## Archivos

- `model.json` - Definición del modelo
- `group1-shard*.bin` - Pesos del modelo
- `vocab.json` - Vocabulario
- `tokenizer.model` - Tokenizer SentencePiece

## Modelo original

[ULFBERTO/OxideLLM_5M](https://huggingface.co/ULFBERTO/OxideLLM_5M)
"""
    
    readme_path = os.path.join(export_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    
    # Crear/actualizar repo
    try:
        create_repo(repo_id, exist_ok=True)
    except:
        pass
    
    # Subir
    print(f"📤 Subiendo TFJS a {repo_id}...")
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update OxideLLM TFJS model"
    )
    
    print(f"✅ Modelo TFJS subido: https://huggingface.co/{repo_id}")


# Uso completo:
# tfjs_dir = convert_to_tfjs(
#     saved_model_path=os.path.join(OUTPUT_PATH, "hf_export/saved_model"),
#     output_path=OUTPUT_PATH
# )
# 
# upload_tfjs_to_huggingface(
#     tfjs_dir=tfjs_dir,
#     tokenizer_path=TOKENIZER_PATH,
#     model_config=model.get_config(),
#     repo_id="ULFBERTO/OxideLLM_5M-tfjs",
#     hf_token="hf_xxxxx"
# )
