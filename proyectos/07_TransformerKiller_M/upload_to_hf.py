"""
Script para subir OxideLLM_TK_SSM_V1 a HuggingFace Hub

Requisitos:
    pip install huggingface_hub

Uso:
    1. Crea una cuenta en https://huggingface.co
    2. Crea un token en https://huggingface.co/settings/tokens (con permisos de escritura)
    3. Ejecuta: huggingface-cli login
    4. Ejecuta: python upload_to_hf.py
"""

import os
import torch
from huggingface_hub import HfApi, create_repo, upload_file
import json

# Configuración
MODEL_NAME = "OxideLLM_TK_SSM_V1"
CHECKPOINT_PATH = "ssm_checkpoint.pth"

def create_model_card():
    """Crea el README.md para HuggingFace"""
    return """---
license: mit
language:
- es
- en
tags:
- ssm
- state-space-model
- mamba-like
- text-generation
- experimental
---

# OxideLLM_TK_SSM_V1

🦀 **Transformer Killer** - Un modelo experimental basado en State Space Models (SSM)

## Descripción

Este modelo utiliza una arquitectura **SSM (State Space Model)** inspirada en Mamba, 
que reemplaza el mecanismo de atención de los Transformers tradicionales con un 
escaneo secuencial selectivo de complejidad **O(n) lineal**.

### Características

- **Arquitectura**: SSM Selectivo (Mamba-like)
- **Parámetros**: ~770K
- **Tokenizer**: Nivel de carácter (228 tokens)
- **Contexto**: Teóricamente ilimitado (complejidad lineal)
- **Entrenamiento**: Iter 1200+

### Ventajas del SSM sobre Transformers

| Aspecto | Transformer | SSM |
|---------|-------------|-----|
| Complejidad | O(n²) | O(n) |
| Memoria | Crece cuadráticamente | Crece linealmente |
| Contexto largo | Costoso | Eficiente |

## Uso

```python
import torch
from model import TransformerKiller
from tokenizer import CharacterTokenizer

# Cargar checkpoint
cp = torch.load("ssm_checkpoint.pth", map_location="cpu")

# Reconstruir tokenizer
tokenizer = CharacterTokenizer()
tokenizer.chars = cp['tokenizer_chars']
tokenizer.vocab_size = len(tokenizer.chars)
tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.chars)}

# Cargar modelo
model = TransformerKiller(
    vocab_size=tokenizer.vocab_size,
    dim=128,
    n_layers=4,
    state_dim=16
)
model.load_state_dict(cp['model_state_dict'])
model.eval()

# Generar texto
def generate(prompt, max_tokens=100):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(idx)[:, -1, :]
            probs = torch.softmax(logits / 0.8, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    return tokenizer.decode(idx[0].tolist())

print(generate("Hola"))
```

## Archivos

- `ssm_checkpoint.pth` - Checkpoint del modelo (pesos + tokenizer)
- `model.py` - Arquitectura SSM
- `tokenizer.py` - Tokenizer a nivel de carácter
- `chat.py` - Script de chat interactivo

## Limitaciones

⚠️ Este es un modelo **experimental y educativo** con solo ~770K parámetros.
No está diseñado para uso en producción. Las respuestas pueden ser incoherentes.

## Licencia

MIT License

## Autor

Entrenado con 🔥 usando PyTorch + CUDA
"""

def create_config():
    """Crea config.json con los hiperparámetros"""
    return {
        "model_type": "ssm_transformer_killer",
        "vocab_size": 228,
        "dim": 128,
        "n_layers": 4,
        "state_dim": 16,
        "architecture": "State Space Model (Mamba-like)",
        "tokenizer_type": "character-level"
    }

def main():
    print(f"Preparando {MODEL_NAME} para HuggingFace...")
    
    # Verificar que existe el checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: No se encontró {CHECKPOINT_PATH}")
        return
    
    # Cargar checkpoint para verificar
    cp = torch.load(CHECKPOINT_PATH, map_location="cpu")
    print(f"Checkpoint cargado - Iter: {cp.get('iter', '?')}")
    
    # Crear archivos temporales
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(create_model_card())
    print("README.md creado")
    
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(create_config(), f, indent=2)
    print("config.json creado")
    
    # Subir a HuggingFace
    api = HfApi()
    
    try:
        # Crear repositorio (cambia TU_USUARIO por tu username de HF)
        repo_id = api.whoami()["name"] + "/" + MODEL_NAME
        
        print(f"\nCreando repositorio: {repo_id}")
        create_repo(repo_id, exist_ok=True)
        
        # Subir archivos
        files_to_upload = [
            ("ssm_checkpoint.pth", "ssm_checkpoint.pth"),
            ("model.py", "model.py"),
            ("tokenizer.py", "tokenizer.py"),
            ("chat.py", "chat.py"),
            ("README.md", "README.md"),
            ("config.json", "config.json"),
        ]
        
        for local_path, repo_path in files_to_upload:
            if os.path.exists(local_path):
                print(f"Subiendo {local_path}...")
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id
                )
        
        print(f"\n✅ Modelo subido exitosamente!")
        print(f"🔗 https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nAsegúrate de:")
        print("1. Tener instalado: pip install huggingface_hub")
        print("2. Estar logueado: huggingface-cli login")

if __name__ == "__main__":
    main()
