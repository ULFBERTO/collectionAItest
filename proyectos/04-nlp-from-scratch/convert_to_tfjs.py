"""
Exporta los pesos del modelo para usar con TensorFlow.js.
Dado los problemas de compatibilidad con Keras 3, exportamos los pesos en formato JSON/binario.

Uso:
    python convert_to_tfjs.py
    python convert_to_tfjs.py --upload
"""

import os
import json
import shutil
import numpy as np

# Importar las clases custom del modelo
from model import GPTModel, PositionalEmbedding, CausalSelfAttention, FeedForward, TransformerBlock

# Rutas
MODEL_DIR = "./gpt_don_quijote_saved_model"
TFJS_OUTPUT_DIR = "./tfjs_model"
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")


def convert_model():
    """Exporta el modelo en formato compatible con TFJS"""
    import tensorflow as tf
    
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"❌ No se encontró el modelo en {KERAS_MODEL_PATH}")
        return False
    
    os.makedirs(TFJS_OUTPUT_DIR, exist_ok=True)
    
    print("📦 Cargando modelo...")
    custom_objects = {
        'GPTModel': GPTModel,
        'PositionalEmbedding': PositionalEmbedding,
        'CausalSelfAttention': CausalSelfAttention,
        'FeedForward': FeedForward,
        'TransformerBlock': TransformerBlock,
    }
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, custom_objects=custom_objects)
    print("✅ Modelo cargado")
    
    # Exportar pesos como archivos binarios
    print("💾 Exportando pesos...")
    weights_manifest = []
    weight_data = []
    
    for i, weight in enumerate(model.weights):
        name = weight.name
        shape = weight.shape.as_list()
        dtype = "float32"
        
        # Obtener datos del peso
        data = weight.numpy().astype(np.float32)
        weight_data.append(data.tobytes())
        
        weights_manifest.append({
            "name": name,
            "shape": shape,
            "dtype": dtype
        })
    
    # Guardar pesos en un archivo binario
    weights_bin_path = os.path.join(TFJS_OUTPUT_DIR, "weights.bin")
    with open(weights_bin_path, "wb") as f:
        for data in weight_data:
            f.write(data)
    
    # Calcular offsets para cada peso
    offset = 0
    for i, manifest in enumerate(weights_manifest):
        size = int(np.prod(manifest["shape"])) * 4  # float32 = 4 bytes
        manifest["offset"] = int(offset)
        manifest["size"] = int(size)
        offset += size
    
    # Crear model.json con la configuración
    model_config = {
        "format": "weights_only",
        "modelConfig": {
            "vocab_size": 110,
            "d_model": 256,
            "num_heads": 4,
            "dff": 512,
            "num_layers": 4,
            "max_len": 100,
            "dropout": 0.1
        },
        "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": weights_manifest
        }]
    }
    
    model_json_path = os.path.join(TFJS_OUTPUT_DIR, "model.json")
    with open(model_json_path, "w") as f:
        json.dump(model_config, f, indent=2)
    
    print(f"✅ model.json creado")
    print(f"✅ weights.bin creado ({offset / 1024 / 1024:.2f} MB)")
    
    # Copiar vocab y config
    copy_metadata()
    
    print(f"\n🎉 ¡Exportación completada!")
    print(f"📁 Archivos en: {TFJS_OUTPUT_DIR}/")
    
    return True


def copy_metadata():
    """Copia vocab.json, config.json"""
    vocab_src = os.path.join(MODEL_DIR, "vocab.json")
    config_src = os.path.join(MODEL_DIR, "config.json")
    
    if os.path.exists(vocab_src):
        shutil.copy(vocab_src, os.path.join(TFJS_OUTPUT_DIR, "vocab.json"))
        print("✅ vocab.json copiado")
    
    if os.path.exists(config_src):
        shutil.copy(config_src, os.path.join(TFJS_OUTPUT_DIR, "config.json"))
        print("✅ config.json copiado")


def upload_to_hf():
    """Sube el modelo a HuggingFace"""
    if not os.path.exists(TFJS_OUTPUT_DIR):
        print("❌ Primero convierte el modelo")
        return
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub no instalado")
        return
    
    REPO_ID = "ULFBERTO/gpt-don-quijote-tfjs"
    
    print(f"📦 Creando repositorio {REPO_ID}...")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Nota: {e}")
    
    # Crear README
    readme = """# GPT Don Quijote - TensorFlow.js

Modelo GPT entrenado con el texto de Don Quijote de la Mancha.
Exportado para usar en navegador con TensorFlow.js.

## Archivos

- `model.json` - Configuracion y manifest de pesos
- `weights.bin` - Pesos del modelo en binario
- `vocab.json` - Vocabulario (char2idx, idx2char)
- `config.json` - Configuracion del modelo

## Uso

Este modelo requiere una implementacion custom del modelo GPT en JavaScript.
Ver el proyecto ZeroCloud para un ejemplo de implementacion.
"""
    
    readme_path = os.path.join(TFJS_OUTPUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    
    print(f"⬆️ Subiendo a HuggingFace...")
    api = HfApi()
    api.upload_folder(
        folder_path=TFJS_OUTPUT_DIR,
        repo_id=REPO_ID,
        repo_type="model"
    )
    
    print(f"🎉 ¡Subido exitosamente!")
    print(f"🔗 URL: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    
    success = convert_model()
    
    if success and args.upload:
        print("\n" + "="*50)
        upload_to_hf()
