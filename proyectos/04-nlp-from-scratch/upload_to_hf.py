"""Script para subir el modelo OxideLLM a Hugging Face Hub."""

import os
import json
from huggingface_hub import HfApi, create_repo

# Configuración
REPO_ID = "ULFBERTO/OxideLLM_5M"
MODEL_DIR = "./OxideLLM_5M_saved_model"


def create_readme(config: dict, vocab_size: int) -> str:
    """Crea el README.md para el repositorio."""
    total_params = estimate_params(config)
    
    return f"""---
license: mit
language:
- es
tags:
- text-generation
- tensorflow
- spanish
- gpt
- transformer
library_name: tensorflow
---

# OxideLLM

Modelo de lenguaje en español entrenado desde cero con literatura clásica española.

## Arquitectura

- **Tipo**: Transformer decoder-only (GPT-like)
- **Parámetros**: ~{total_params/1e6:.1f}M
- **Vocabulario**: {vocab_size} tokens (BPE con SentencePiece)
- **Contexto máximo**: {config.get('max_len', 512)} tokens
- **Capas**: {config.get('num_layers', 6)}
- **Dimensión**: {config.get('d_model', 512)}
- **Cabezas de atención**: {config.get('num_heads', 8)}
- **FFN dimensión**: {config.get('dff', 2048)}

## Corpus de entrenamiento

Entrenado con literatura clásica española incluyendo:
- Don Quijote de la Mancha
- La Celestina
- Lazarillo de Tormes
- Novelas de Galdós (Fortunata y Jacinta, Marianela, etc.)
- La Regenta
- Rimas y Leyendas de Bécquer
- Y más...

## Uso

```python
import tensorflow as tf
import sentencepiece as spm
from huggingface_hub import snapshot_download

# Descargar modelo
model_path = snapshot_download(repo_id="{REPO_ID}")

# Cargar tokenizer
sp = spm.SentencePieceProcessor()
sp.load(f"{{model_path}}/tokenizer.model")

# Cargar modelo
model = tf.keras.models.load_model(f"{{model_path}}/saved_model")

# Generar texto
def generate(prompt, max_tokens=100, temperature=0.8):
    tokens = sp.encode(prompt)
    for _ in range(max_tokens):
        input_ids = tf.constant([tokens[-512:]], dtype=tf.int32)
        logits = model(input_ids, training=False)
        logits = logits[0, -1, :] / temperature
        
        top_k = 40
        top_logits, top_indices = tf.math.top_k(logits, k=top_k)
        probs = tf.nn.softmax(top_logits)
        next_idx = tf.random.categorical(tf.expand_dims(tf.math.log(probs), 0), 1)
        next_token = top_indices[next_idx[0, 0]].numpy()
        
        tokens.append(int(next_token))
        if next_token == 3:  # </s>
            break
    
    return sp.decode(tokens)

# Ejemplo
text = generate("En un lugar de la Mancha")
print(text)
```

## Archivos

- `saved_model/` - Modelo TensorFlow SavedModel
- `model_weights.h5` - Pesos en formato H5
- `tokenizer.model` - Tokenizer SentencePiece
- `tokenizer.vocab` - Vocabulario del tokenizer
- `vocab.json` - Vocabulario en formato JSON
- `config.json` - Configuración del modelo
- `training_history.json` - Historial de entrenamiento

## Licencia

MIT
"""


def estimate_params(config: dict) -> int:
    """Estima el número de parámetros del modelo."""
    vocab_size = config.get('vocab_size', 8000)
    d_model = config.get('d_model', 512)
    num_layers = config.get('num_layers', 6)
    dff = config.get('dff', 2048)
    
    # Embedding
    params = vocab_size * d_model
    # Positional encoding (learned)
    params += config.get('max_len', 512) * d_model
    # Transformer blocks
    for _ in range(num_layers):
        # Self-attention
        params += 4 * d_model * d_model  # Q, K, V, O
        # FFN
        params += d_model * dff + dff * d_model
        # Layer norms
        params += 4 * d_model
    # Final layer norm + output
    params += d_model + vocab_size * d_model
    
    return params


def upload(repo_id: str = None, model_dir: str = None, token: str = None):
    """Sube el modelo a Hugging Face Hub."""
    repo_id = repo_id or REPO_ID
    model_dir = model_dir or MODEL_DIR
    
    # Verificar que existe el modelo
    if not os.path.exists(model_dir):
        print(f"❌ No se encontró el directorio {model_dir}")
        return False
    
    # Verificar archivos necesarios
    required_files = ["config.json"]
    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            print(f"❌ No se encontró {f} en {model_dir}")
            return False
    
    # Cargar configuración
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    vocab_size = config.get("vocab_size", 8000)
    
    # Crear README
    readme_content = create_readme(config, vocab_size)
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✅ README.md creado")
    
    # Crear repositorio
    print(f"📦 Creando repositorio {repo_id}...")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception as e:
        print(f"Nota: {e}")
    
    # Subir
    print(f"⬆️ Subiendo modelo desde {model_dir}...")
    api = HfApi()
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    
    print(f"🎉 ¡Modelo subido exitosamente!")
    print(f"🔗 URL: https://huggingface.co/{repo_id}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Subir OxideLLM a Hugging Face")
    parser.add_argument("--repo", type=str, default=REPO_ID, help="Repo ID")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR, help="Directorio del modelo")
    parser.add_argument("--token", type=str, default=None, help="Token de HF")
    args = parser.parse_args()
    
    upload(repo_id=args.repo, model_dir=args.model_dir, token=args.token)
