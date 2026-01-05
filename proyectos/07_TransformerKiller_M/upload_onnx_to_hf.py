"""
Script para subir OxideLLM_TK_SSM_V1_ONNX a HuggingFace Hub

Uso:
    python upload_onnx_to_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file

MODEL_NAME = "OxideLLM_TK_SSM_V1_ONNX"

def create_model_card():
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
- onnx
- onnxruntime
- browser
- experimental
---

# OxideLLM_TK_SSM_V1_ONNX

🦀 **Transformer Killer** - Versión ONNX para navegador

## ⚠️ Versión ONNX

Esta es la versión **ONNX** del modelo SSM, convertida para ejecutarse en navegadores web 
usando ONNX Runtime Web. La versión original en PyTorch está disponible en 
[OxideLLM_TK_SSM_V1](https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1).

> **Nota**: Actualmente no hay soporte nativo para PyTorch en navegadores, por lo que 
> se requiere esta conversión a ONNX para uso web.

## Descripción

Modelo experimental basado en **State Space Models (SSM)** inspirado en Mamba, 
que reemplaza el mecanismo de atención de los Transformers con un escaneo 
secuencial selectivo de complejidad **O(n) lineal**.

### Especificaciones

| Aspecto | Valor |
|---------|-------|
| Arquitectura | SSM Selectivo (Mamba-like) |
| Parámetros | ~770K |
| Formato | ONNX |
| Tamaño | ~4 MB |
| Complejidad | O(n) lineal |
| Tokenizer | Nivel de carácter |

## Uso en Navegador (JavaScript)

```javascript
import * as ort from 'onnxruntime-web';

// Cargar modelo
const session = await ort.InferenceSession.create('ssm_model.onnx');

// Cargar tokenizer
const tokenizer = await fetch('tokenizer.json').then(r => r.json());

// Codificar texto
const text = "Hola ";
const inputIds = text.split('').map(c => tokenizer.char2idx[c] || 0);

// Inferencia
const tensor = new ort.Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]);
const outputs = await session.run({ input_ids: tensor });
const logits = outputs.logits.data;

// Decodificar siguiente token...
```

## Archivos

- `ssm_model.onnx` - Modelo ONNX (~4 MB)
- `tokenizer.json` - Vocabulario y configuración

## Limitaciones

⚠️ **Experimental**: 
- Modelo pequeño (~770K params) para propósitos educativos
- La conversión ONNX puede tener diferencias menores vs PyTorch
- Algunas operaciones del SSM pueden no estar optimizadas en ONNX

## Links

- 🔥 [Versión PyTorch](https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1)
- 🌐 [Demo en ZeroCloud](https://zerocloud.vercel.app)

## Licencia

MIT License
"""

def main():
    print(f"Subiendo {MODEL_NAME} a HuggingFace...")
    
    # Verificar archivos
    required_files = ["ssm_model.onnx", "tokenizer.json"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: No se encontró {f}")
            print("Ejecuta primero: python convert_to_onnx.py")
            return
    
    # Crear README
    with open("README_ONNX.md", "w", encoding="utf-8") as f:
        f.write(create_model_card())
    
    api = HfApi()
    
    try:
        repo_id = api.whoami()["name"] + "/" + MODEL_NAME
        print(f"Creando repositorio: {repo_id}")
        create_repo(repo_id, exist_ok=True)
        
        # Subir archivos
        files = [
            ("ssm_model.onnx", "ssm_model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("README_ONNX.md", "README.md"),
        ]
        
        for local, remote in files:
            if os.path.exists(local):
                print(f"Subiendo {local}...")
                upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo_id)
        
        print(f"\n✅ Modelo ONNX subido!")
        print(f"🔗 https://huggingface.co/{repo_id}")
        
        # Limpiar
        os.remove("README_ONNX.md")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
