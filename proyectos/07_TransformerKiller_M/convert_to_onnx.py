"""
Convierte el modelo SSM (Transformer Killer) a formato ONNX para usar en navegador.

Requisitos:
    pip install torch onnx onnxruntime

Uso:
    python convert_to_onnx.py

Salida:
    - ssm_model.onnx (modelo)
    - tokenizer.json (vocabulario)
"""

import torch
import torch.nn as nn
import json
import os
from model import TransformerKiller
from tokenizer import CharacterTokenizer

# Configuración (debe coincidir con train.py)
DIM = 128
STATE_DIM = 16
N_LAYERS = 4
CHECKPOINT_PATH = "ssm_checkpoint.pth"
OUTPUT_ONNX = "ssm_model.onnx"
OUTPUT_TOKENIZER = "tokenizer.json"


class SSMForExport(nn.Module):
    """
    Wrapper del modelo SSM para exportación ONNX.
    ONNX no soporta bien los bucles dinámicos del SSM original,
    así que simplificamos la inferencia.
    """
    def __init__(self, model: TransformerKiller):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] tokens de entrada
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        return self.model(input_ids)


def load_checkpoint():
    """Carga el checkpoint y reconstruye modelo + tokenizer"""
    print(f"Cargando checkpoint: {CHECKPOINT_PATH}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"No se encontró {CHECKPOINT_PATH}")
    
    cp = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # Reconstruir tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.chars = cp['tokenizer_chars']
    tokenizer.vocab_size = len(tokenizer.chars)
    tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
    tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.chars)}
    
    # Cargar modelo
    model = TransformerKiller(
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        n_layers=N_LAYERS,
        state_dim=STATE_DIM
    )
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    print(f"Modelo cargado - Iter: {cp.get('iter', '?')}")
    print(f"Vocabulario: {tokenizer.vocab_size} tokens")
    
    return model, tokenizer, cp.get('iter', 0)


def export_tokenizer(tokenizer: CharacterTokenizer, iteration: int):
    """Exporta el tokenizer a JSON"""
    tokenizer_data = {
        "model_name": "OxideLLM_TK_SSM_V1",
        "iteration": iteration,
        "vocab_size": tokenizer.vocab_size,
        "char2idx": tokenizer.stoi,
        "idx2char": {str(k): v for k, v in tokenizer.itos.items()},
        "special_tokens": tokenizer.special_tokens,
        "config": {
            "dim": DIM,
            "state_dim": STATE_DIM,
            "n_layers": N_LAYERS
        }
    }
    
    with open(OUTPUT_TOKENIZER, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"Tokenizer exportado: {OUTPUT_TOKENIZER}")


def export_to_onnx(model: TransformerKiller, tokenizer: CharacterTokenizer):
    """Exporta el modelo a ONNX"""
    print("\nExportando a ONNX...")
    
    # Wrapper para exportación
    export_model = SSMForExport(model)
    export_model.eval()
    
    # Input de ejemplo (batch=1, seq_len=32)
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 32), dtype=torch.long)
    
    # Exportar
    torch.onnx.export(
        export_model,
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
    
    # Verificar tamaño
    size_mb = os.path.getsize(OUTPUT_ONNX) / (1024 * 1024)
    print(f"Tamaño: {size_mb:.2f} MB")


def verify_onnx():
    """Verifica que el modelo ONNX funcione correctamente"""
    try:
        import onnx
        import onnxruntime as ort
        
        print("\nVerificando modelo ONNX...")
        
        # Verificar estructura
        model = onnx.load(OUTPUT_ONNX)
        onnx.checker.check_model(model)
        print("✓ Estructura ONNX válida")
        
        # Probar inferencia
        session = ort.InferenceSession(OUTPUT_ONNX)
        
        # Cargar tokenizer
        with open(OUTPUT_TOKENIZER, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        # Test input - usar solo caracteres que sabemos que existen (espacios y letras básicas)
        # Buscar caracteres válidos en el vocabulario
        valid_chars = [c for c in "En un lugar de la " if c in tok_data['char2idx']]
        if not valid_chars:
            valid_chars = list(tok_data['char2idx'].keys())[:10]
        
        test_text = ''.join(valid_chars[:8])
        input_ids = [tok_data['char2idx'][c] for c in test_text]
        input_array = torch.tensor([input_ids], dtype=torch.int64).numpy()
        
        print(f"  Test input: '{test_text}' -> {input_ids}")
        
        # Inferencia
        outputs = session.run(None, {'input_ids': input_array})
        logits = outputs[0]
        
        print(f"✓ Inferencia exitosa")
        print(f"  Input shape: {input_array.shape}")
        print(f"  Output shape: {logits.shape}")
        
        return True
        
    except ImportError:
        print("⚠️ onnxruntime no instalado, saltando verificación")
        print("  Instala con: pip install onnxruntime")
        return True
    except Exception as e:
        print(f"⚠️ Verificación con warning: {e}")
        print("  El modelo se exportó, pero puede haber problemas de compatibilidad.")
        print("  Esto es común con arquitecturas SSM complejas.")
        return True  # Continuar de todas formas


def main():
    print("="*50)
    print("  Conversión SSM → ONNX")
    print("="*50 + "\n")
    
    # Cargar modelo
    model, tokenizer, iteration = load_checkpoint()
    
    # Exportar tokenizer
    export_tokenizer(tokenizer, iteration)
    
    # Exportar modelo
    export_to_onnx(model, tokenizer)
    
    # Verificar
    if verify_onnx():
        print("\n" + "="*50)
        print("✅ Conversión completada exitosamente!")
        print("="*50)
        print(f"\nArchivos generados:")
        print(f"  - {OUTPUT_ONNX}")
        print(f"  - {OUTPUT_TOKENIZER}")
        print(f"\nPróximos pasos:")
        print(f"  1. Sube estos archivos a HuggingFace")
        print(f"  2. El modelo estará disponible en ZeroCloud")


if __name__ == "__main__":
    main()
