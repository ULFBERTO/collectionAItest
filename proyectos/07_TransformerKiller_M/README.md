---
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
