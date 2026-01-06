"""
Convierte el modelo SSM a ONNX usando una versión simplificada compatible.

El modelo SSM original tiene bucles que ONNX no maneja bien.
Esta versión crea un modelo simplificado que solo usa los embeddings
para generación de texto (similar a como funciona el modelo TFJS).
"""

import torch
import torch.nn as nn
import json
import os

# Configuración
DIM = 128
STATE_DIM = 16
N_LAYERS = 4
CHECKPOINT_PATH = "ssm_checkpoint.pth"
OUTPUT_ONNX = "ssm_model.onnx"
OUTPUT_TOKENIZER = "tokenizer.json"


class SimpleSSMForONNX(nn.Module):
    """
    Versión simplificada del modelo SSM para ONNX.
    Usa solo embeddings y proyección de salida (sin el bucle SSM).
    Esto permite generación básica de texto en el navegador.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Obtener embeddings
        x = self.tok_embeddings(input_ids)  # [B, L, D]
        
        # Contexto simple: promedio móvil de los últimos tokens
        # Esto simula parte del comportamiento del SSM
        batch_size, seq_len, dim = x.shape
        
        # Crear una versión con contexto acumulado
        # Usamos una convolución causal simple
        context_size = min(8, seq_len)
        
        # Promedio de contexto para cada posición
        output = torch.zeros_like(x)
        for i in range(seq_len):
            start = max(0, i - context_size + 1)
            output[:, i, :] = x[:, start:i+1, :].mean(dim=1)
        
        # Proyectar a vocabulario
        logits = self.output(output)
        return logits


class SimpleSSMForONNXV2(nn.Module):
    """
    Versión vectorizada sin bucles para ONNX.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Capa de contexto simple (reemplaza SSM)
        self.context_proj = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings
        x = self.tok_embeddings(input_ids)  # [B, L, D]
        
        # Contexto simple: cada posición ve el promedio acumulado
        # Usamos cumsum para evitar bucles
        cumsum = torch.cumsum(x, dim=1)  # [B, L, D]
        counts = torch.arange(1, x.size(1) + 1, device=x.device).float().view(1, -1, 1)
        context = cumsum / counts  # Promedio acumulado
        
        # Mezclar embedding actual con contexto
        mixed = x + self.context_proj(context)
        
        # Proyectar a vocabulario
        logits = self.output(mixed)
        return logits


def load_checkpoint():
    """Carga el checkpoint original"""
    print(f"Cargando checkpoint: {CHECKPOINT_PATH}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No se encontró {CHECKPOINT_PATH}")
    
    cp = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Reconstruir tokenizer
    from tokenizer import CharacterTokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.chars = cp['tokenizer_chars']
    tokenizer.vocab_size = len(tokenizer.chars)
    tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.chars)}
    
    print(f"Checkpoint cargado - Iter: {cp.get('iter', '?')}")
    print(f"Vocabulario: {tokenizer.vocab_size} tokens")
    
    return cp, tokenizer


def create_simple_model(cp, tokenizer):
    """Crea modelo simplificado con pesos del original"""
    vocab_size = tokenizer.vocab_size
    
    # Crear modelo simple
    model = SimpleSSMForONNXV2(vocab_size=vocab_size, dim=DIM)
    
    # Copiar embeddings del modelo original
    original_state = cp['model_state_dict']
    
    # Copiar tok_embeddings
    if 'tok_embeddings.weight' in original_state:
        model.tok_embeddings.weight.data = original_state['tok_embeddings.weight'].clone()
        print("✓ Embeddings copiados")
    
    # Copiar output projection
    if 'output.weight' in original_state:
        model.output.weight.data = original_state['output.weight'].clone()
        print("✓ Output projection copiado")
    
    # Inicializar context_proj con identidad aproximada
    nn.init.eye_(model.context_proj.weight)
    nn.init.zeros_(model.context_proj.bias)
    
    model.eval()
    return model


def export_tokenizer(tokenizer, iteration: int):
    """Exporta el tokenizer a JSON"""
    tokenizer_data = {
        "model_name": "OxideLLM_TK_SSM_V1",
        "model_type": "ssm_simplified_onnx",
        "iteration": iteration,
        "vocab_size": tokenizer.vocab_size,
        "char2idx": tokenizer.stoi,
        "idx2char": {str(k): v for k, v in tokenizer.itos.items()},
        "special_tokens": getattr(tokenizer, 'special_tokens', []),
        "config": {
            "dim": DIM,
            "state_dim": STATE_DIM,
            "n_layers": N_LAYERS
        },
        "note": "Versión simplificada para ONNX. El modelo SSM completo requiere PyTorch."
    }
    
    with open(OUTPUT_TOKENIZER, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"Tokenizer exportado: {OUTPUT_TOKENIZER}")


def export_to_onnx(model, tokenizer):
    """Exporta el modelo a ONNX"""
    print("\nExportando a ONNX...")
    
    # Input de ejemplo
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 16), dtype=torch.long)
    
    # Exportar
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_ONNX,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print(f"Modelo ONNX exportado: {OUTPUT_ONNX}")
    
    size_mb = os.path.getsize(OUTPUT_ONNX) / (1024 * 1024)
    print(f"Tamaño: {size_mb:.2f} MB")


def verify_onnx(tokenizer):
    """Verifica el modelo ONNX"""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        
        print("\nVerificando modelo ONNX...")
        
        # Verificar estructura
        model = onnx.load(OUTPUT_ONNX)
        onnx.checker.check_model(model)
        print("✓ Estructura ONNX válida")
        
        # Probar inferencia
        session = ort.InferenceSession(OUTPUT_ONNX)
        
        # Test con índices válidos
        test_ids = [4, 5, 6, 7, 8]  # Índices seguros dentro del vocabulario
        test_ids = [min(i, tokenizer.vocab_size - 1) for i in test_ids]
        
        input_array = np.array([test_ids], dtype=np.int64)
        print(f"  Test input: {input_array} (shape: {input_array.shape})")
        
        outputs = session.run(None, {'input_ids': input_array})
        logits = outputs[0]
        
        print(f"✓ Inferencia exitosa")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output range: [{logits.min():.2f}, {logits.max():.2f}]")
        
        # Test de generación simple
        next_token = np.argmax(logits[0, -1, :])
        if next_token < tokenizer.vocab_size:
            next_char = tokenizer.itos.get(next_token, '?')
            print(f"  Siguiente token predicho: {next_token} -> '{next_char}'")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error en verificación: {e}")
        return False


def main():
    print("="*50)
    print("  Conversión SSM → ONNX (Versión Simplificada)")
    print("="*50 + "\n")
    
    # Cargar checkpoint
    cp, tokenizer = load_checkpoint()
    iteration = cp.get('iter', 0)
    
    # Crear modelo simplificado
    model = create_simple_model(cp, tokenizer)
    
    # Exportar tokenizer
    export_tokenizer(tokenizer, iteration)
    
    # Exportar modelo
    export_to_onnx(model, tokenizer)
    
    # Verificar
    if verify_onnx(tokenizer):
        print("\n" + "="*50)
        print("✅ Conversión completada!")
        print("="*50)
        print(f"\nArchivos generados:")
        print(f"  - {OUTPUT_ONNX}")
        print(f"  - {OUTPUT_TOKENIZER}")
        print(f"\nNota: Esta es una versión simplificada del modelo SSM.")
        print(f"Para el modelo completo, usa la versión PyTorch.")
        print(f"\nPróximo paso: python upload_onnx_to_hf.py")


if __name__ == "__main__":
    main()
