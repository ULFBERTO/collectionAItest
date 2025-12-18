"""
Utilidades para integrar el modelo con Hugging Face Hub.
Permite subir y descargar modelos entrenados.

Instalación requerida:
    pip install huggingface_hub

Configuración inicial (solo una vez):
    huggingface-cli login
    # O usar token directamente con HF_TOKEN en variables de entorno
"""

import os
import json
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ huggingface_hub no instalado. Ejecuta: pip install huggingface_hub")


# Configuración por defecto
DEFAULT_REPO_ID = "ULFBERTO/gpt-don-quijote"
MODEL_DIR = "./gpt_don_quijote_saved_model"
CHECKPOINT_DIR = "./training_checkpoints"


def check_hf_available():
    """Verifica si huggingface_hub está disponible."""
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub no está instalado. "
            "Ejecuta: pip install huggingface_hub"
        )


def create_model_card(repo_id: str, config: dict) -> str:
    """Crea un README.md (Model Card) para el repositorio."""
    return f"""---
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

- **Vocabulario**: {config.get('vocab_size', 'N/A')} caracteres
- **Dimensión del modelo (d_model)**: {config.get('d_model', 256)}
- **Número de cabezas de atención**: {config.get('num_heads', 4)}
- **Dimensión feedforward (dff)**: {config.get('dff', 512)}
- **Número de capas**: {config.get('num_layers', 4)}
- **Longitud máxima de secuencia**: {config.get('max_len', 100)}

## Uso

```python
from huggingface_hub import snapshot_download
import tensorflow as tf
import json

# Descargar modelo
model_path = snapshot_download(repo_id="{repo_id}")

# Cargar vocabulario
with open(f"{{model_path}}/vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
    
char2idx = vocab_data["char2idx"]
idx2char = {{int(k): v for k, v in vocab_data["idx2char"].items()}}

# Cargar modelo
model = tf.keras.models.load_model(f"{{model_path}}/saved_model")
```

## Entrenamiento

Entrenado con TensorFlow 2.x usando arquitectura Transformer (decoder-only).
"""


def upload_model(
    repo_id: str = None,
    model_dir: str = MODEL_DIR,
    vocab_data: dict = None,
    config: dict = None,
    private: bool = False,
    token: str = None
):
    """
    Sube el modelo entrenado a Hugging Face Hub.
    
    Args:
        repo_id: ID del repositorio (usuario/nombre-modelo)
        model_dir: Directorio con el modelo guardado
        vocab_data: Diccionario con char2idx e idx2char
        config: Configuración del modelo
        private: Si el repositorio debe ser privado
        token: Token de HF (opcional, usa el de huggingface-cli login si no se proporciona)
    """
    check_hf_available()
    
    repo_id = repo_id or DEFAULT_REPO_ID
    if repo_id == "tu-usuario/gpt-don-quijote":
        raise ValueError(
            "Debes configurar tu repo_id. Edita DEFAULT_REPO_ID en huggingface_utils.py "
            "o pasa repo_id como parámetro."
        )
    
    api = HfApi(token=token)
    
    # Crear repositorio si no existe
    print(f"📦 Creando/verificando repositorio: {repo_id}")
    try:
        create_repo(repo_id, repo_type="model", private=private, token=token, exist_ok=True)
    except Exception as e:
        print(f"Nota: {e}")
    
    # Guardar vocabulario como JSON
    if vocab_data:
        vocab_path = Path(model_dir) / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Vocabulario guardado en {vocab_path}")
    
    # Guardar configuración
    if config:
        config_path = Path(model_dir) / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuración guardada en {config_path}")
    
    # Crear Model Card
    model_card = create_model_card(repo_id, config or {})
    readme_path = Path(model_dir) / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"✅ Model Card creado en {readme_path}")
    
    # Subir todo el directorio
    print(f"⬆️ Subiendo modelo a {repo_id}...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )
    
    print(f"🎉 ¡Modelo subido exitosamente!")
    print(f"🔗 URL: https://huggingface.co/{repo_id}")
    
    return f"https://huggingface.co/{repo_id}"


def download_model(
    repo_id: str = None,
    local_dir: str = "./downloaded_model",
    token: str = None
):
    """
    Descarga un modelo desde Hugging Face Hub.
    
    Args:
        repo_id: ID del repositorio (usuario/nombre-modelo)
        local_dir: Directorio local donde guardar
        token: Token de HF (para repos privados)
    
    Returns:
        Ruta al directorio descargado
    """
    check_hf_available()
    
    repo_id = repo_id or DEFAULT_REPO_ID
    
    print(f"⬇️ Descargando modelo desde {repo_id}...")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token
    )
    
    print(f"✅ Modelo descargado en: {path}")
    return path


def load_from_hub(repo_id: str = None, token: str = None):
    """
    Descarga y carga el modelo desde Hugging Face Hub.
    
    Returns:
        tuple: (model, char2idx, idx2char)
    """
    import tensorflow as tf
    
    check_hf_available()
    
    repo_id = repo_id or DEFAULT_REPO_ID
    
    # Descargar
    model_path = download_model(repo_id, token=token)
    
    # Cargar vocabulario
    vocab_path = Path(model_path) / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    
    char2idx = vocab_data["char2idx"]
    idx2char = {int(k): v for k, v in vocab_data["idx2char"].items()}
    
    # Cargar modelo (Keras 3 usa .keras)
    saved_model_path = Path(model_path) / "model.keras"
    if not saved_model_path.exists():
        # Fallback a formato antiguo
        saved_model_path = Path(model_path) / "saved_model"
        if not saved_model_path.exists():
            saved_model_path = model_path
    
    print(f"🔄 Cargando modelo desde {saved_model_path}...")
    model = tf.keras.models.load_model(str(saved_model_path))
    
    print("✅ Modelo cargado exitosamente")
    return model, char2idx, idx2char


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gestión de modelos en Hugging Face Hub")
    parser.add_argument("action", choices=["upload", "download", "load"], 
                       help="Acción a realizar")
    parser.add_argument("--repo", type=str, default=None,
                       help="ID del repositorio (usuario/nombre-modelo)")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR,
                       help="Directorio del modelo")
    parser.add_argument("--private", action="store_true",
                       help="Crear repositorio privado")
    parser.add_argument("--token", type=str, default=None,
                       help="Token de Hugging Face")
    
    args = parser.parse_args()
    
    if args.action == "upload":
        # Para upload necesitamos cargar el modelo primero para obtener vocab
        from generate import load_model
        model, char2idx, idx2char = load_model()
        
        if model is None:
            print("❌ No se encontró modelo entrenado. Ejecuta train.py primero.")
        else:
            vocab_data = {
                "char2idx": char2idx,
                "idx2char": {str(k): v for k, v in idx2char.items()}
            }
            config = {
                "vocab_size": len(char2idx),
                "d_model": 256,
                "num_heads": 4,
                "dff": 512,
                "num_layers": 4,
                "max_len": 100
            }
            
            # Primero exportar el modelo
            from export_model import export
            export()
            
            # Luego subir
            upload_model(
                repo_id=args.repo,
                model_dir=args.model_dir,
                vocab_data=vocab_data,
                config=config,
                private=args.private,
                token=args.token
            )
    
    elif args.action == "download":
        download_model(repo_id=args.repo, local_dir="./downloaded_model", token=args.token)
    
    elif args.action == "load":
        model, char2idx, idx2char = load_from_hub(repo_id=args.repo, token=args.token)
        print(f"Vocabulario: {len(char2idx)} caracteres")
