"""
Convierte el modelo OxideLLM a formato TensorFlow.js para usar en navegador.

Uso:
    python convert_to_tfjs.py
    python convert_to_tfjs.py --upload
"""

import os
import json
import shutil
import subprocess
import numpy as np

# Rutas
MODEL_DIR = "./OxideLLM_5M_saved_model"
TFJS_OUTPUT_DIR = "./tfjs_model"
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, "saved_model")


def convert_model():
    """Convierte el modelo a formato TFJS."""
    
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"❌ No se encontró el modelo en {SAVED_MODEL_PATH}")
        print("   Ejecuta train.py primero para entrenar el modelo.")
        return False
    
    os.makedirs(TFJS_OUTPUT_DIR, exist_ok=True)
    
    print("🔄 Convirtiendo modelo a TensorFlow.js...")
    
    # Intentar conversión con tensorflowjs_converter
    try:
        result = subprocess.run([
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--signature_name=serving_default",
            SAVED_MODEL_PATH,
            TFJS_OUTPUT_DIR
        ], capture_output=True, text=True, check=True)
        print("✅ Conversión exitosa (tf_saved_model)")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error con tf_saved_model: {e.stderr}")
        print("Intentando método alternativo (keras)...")
        
        try:
            result = subprocess.run([
                "tensorflowjs_converter",
                "--input_format=keras",
                SAVED_MODEL_PATH,
                TFJS_OUTPUT_DIR
            ], capture_output=True, text=True, check=True)
            print("✅ Conversión exitosa (keras)")
            
        except subprocess.CalledProcessError as e2:
            print(f"❌ Error en conversión: {e2.stderr}")
            print("\nIntentando exportación manual de pesos...")
            return export_weights_manually()
    
    except FileNotFoundError:
        print("❌ tensorflowjs_converter no encontrado.")
        print("   Instala con: pip install tensorflowjs")
        return False
    
    # Copiar archivos adicionales
    copy_metadata()
    
    print(f"\n🎉 ¡Conversión completada!")
    print(f"📁 Archivos en: {TFJS_OUTPUT_DIR}/")
    
    return True


def export_weights_manually():
    """Exporta pesos manualmente si tensorflowjs_converter falla."""
    import tensorflow as tf
    
    print("💾 Exportando pesos manualmente...")
    
    # Cargar modelo
    try:
        model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False
    
    weights_manifest = []
    weight_data = []
    
    for weight in model.weights:
        name = weight.name
        shape = weight.shape.as_list()
        data = weight.numpy().astype(np.float32)
        weight_data.append(data.tobytes())
        
        weights_manifest.append({
            "name": name,
            "shape": shape,
            "dtype": "float32"
        })
    
    # Guardar pesos binarios
    weights_bin_path = os.path.join(TFJS_OUTPUT_DIR, "weights.bin")
    with open(weights_bin_path, "wb") as f:
        for data in weight_data:
            f.write(data)
    
    # Calcular offsets
    offset = 0
    for manifest in weights_manifest:
        size = int(np.prod(manifest["shape"])) * 4
        manifest["offset"] = int(offset)
        manifest["size"] = int(size)
        offset += size
    
    # Cargar config del modelo
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
    else:
        model_config = {}
    
    # Crear model.json
    model_json = {
        "format": "weights_only",
        "modelConfig": model_config,
        "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": weights_manifest
        }]
    }
    
    model_json_path = os.path.join(TFJS_OUTPUT_DIR, "model.json")
    with open(model_json_path, "w") as f:
        json.dump(model_json, f, indent=2)
    
    print(f"✅ model.json creado")
    print(f"✅ weights.bin creado ({offset / 1024 / 1024:.2f} MB)")
    
    copy_metadata()
    
    print(f"\n🎉 ¡Exportación completada!")
    return True


def copy_metadata():
    """Copia archivos de metadata al directorio TFJS."""
    files_to_copy = [
        ("vocab.json", "vocab.json"),
        ("config.json", "config.json"),
        ("tokenizer.model", "tokenizer.model"),
        ("tokenizer.vocab", "tokenizer.vocab"),
    ]
    
    for src_name, dst_name in files_to_copy:
        src = os.path.join(MODEL_DIR, src_name)
        dst = os.path.join(TFJS_OUTPUT_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"✅ {src_name} copiado")


def create_tfjs_readme(config: dict) -> str:
    """Crea README para el repo TFJS."""
    return f"""---
license: mit
language:
- es
tags:
- text-generation
- tfjs
- browser
- spanish
- transformer
library_name: tensorflowjs
---

# OxideLLM - TensorFlow.js

Versión TensorFlow.js del modelo OxideLLM para ejecutar en navegador.

## Arquitectura

- **Parámetros**: ~{config.get('d_model', 512) * config.get('num_layers', 6) / 1000:.0f}K+ 
- **Vocabulario**: {config.get('vocab_size', 8000)} tokens (BPE)
- **Contexto**: {config.get('max_len', 512)} tokens
- **Capas**: {config.get('num_layers', 6)}

## Archivos

- `model.json` - Configuración y manifest del modelo
- `weights.bin` o `group1-shard*.bin` - Pesos del modelo
- `vocab.json` - Vocabulario para tokenización
- `tokenizer.model` - Tokenizer SentencePiece (opcional)
- `config.json` - Configuración del modelo

## Uso en JavaScript

```javascript
import * as tf from '@tensorflow/tfjs';

// Cargar modelo
const model = await tf.loadGraphModel('model.json');

// O si es formato layers:
// const model = await tf.loadLayersModel('model.json');

// Cargar vocabulario
const vocabResponse = await fetch('vocab.json');
const vocab = await vocabResponse.json();

// Tokenizar (simplificado - usar librería BPE real en producción)
function tokenize(text) {{
    // Implementar tokenización BPE
    return tokens;
}}

// Generar
async function generate(prompt, maxTokens = 50) {{
    let tokens = tokenize(prompt);
    
    for (let i = 0; i < maxTokens; i++) {{
        const input = tf.tensor2d([tokens.slice(-512)], [1, tokens.length]);
        const logits = model.predict(input);
        const lastLogits = logits.slice([0, tokens.length - 1, 0], [1, 1, -1]);
        
        // Sampling
        const probs = tf.softmax(lastLogits.div(0.8));
        const nextToken = tf.multinomial(probs.squeeze(), 1).dataSync()[0];
        
        tokens.push(nextToken);
        if (nextToken === 3) break; // </s>
    }}
    
    return decode(tokens);
}}
```

## Modelo original

[ULFBERTO/OxideLLM_5M](https://huggingface.co/ULFBERTO/OxideLLM_5M)
"""


def upload_to_hf(token: str = None):
    """Sube el modelo TFJS a HuggingFace."""
    if not os.path.exists(TFJS_OUTPUT_DIR):
        print("❌ Primero convierte el modelo con: python convert_to_tfjs.py")
        return False
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub no instalado. Ejecuta: pip install huggingface_hub")
        return False
    
    REPO_ID = "ULFBERTO/OxideLLM_5M-tfjs"
    
    # Cargar config
    config_path = os.path.join(TFJS_OUTPUT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    # Crear README
    readme = create_tfjs_readme(config)
    readme_path = os.path.join(TFJS_OUTPUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    print("✅ README.md creado")
    
    # Crear repo
    print(f"📦 Creando repositorio {REPO_ID}...")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Nota: {e}")
    
    # Subir
    print(f"⬆️ Subiendo a HuggingFace...")
    api = HfApi()
    api.upload_folder(
        folder_path=TFJS_OUTPUT_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        token=token
    )
    
    print(f"🎉 ¡Subido exitosamente!")
    print(f"🔗 URL: https://huggingface.co/{REPO_ID}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convertir OxideLLM a TFJS")
    parser.add_argument("--upload", action="store_true", help="Subir a HuggingFace")
    parser.add_argument("--token", type=str, default=None, help="Token de HF")
    args = parser.parse_args()
    
    success = convert_model()
    
    if success and args.upload:
        print("\n" + "=" * 50)
        upload_to_hf(token=args.token)
