"""
Exporta los pesos del modelo SSM a formato JSON para usar en JavaScript.
"""

import torch
import json
import numpy as np
import os

CHECKPOINT_PATH = "ssm_checkpoint.pth"
OUTPUT_WEIGHTS = "ssm_weights.json"

def main():
    print("Exportando pesos a JSON...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: No se encontró {CHECKPOINT_PATH}")
        return
    
    # Cargar checkpoint
    cp = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = cp['model_state_dict']
    
    # Convertir tensores a listas
    weights = {}
    for key, tensor in state_dict.items():
        weights[key] = tensor.numpy().tolist()
        print(f"  {key}: {tensor.shape}")
    
    # Agregar info del tokenizer
    weights['_tokenizer_chars'] = cp['tokenizer_chars']
    weights['_iteration'] = cp.get('iter', 0)
    weights['_config'] = {
        'dim': 128,
        'state_dim': 16,
        'n_layers': 4,
        'vocab_size': len(cp['tokenizer_chars'])
    }
    
    # Guardar como JSON
    with open(OUTPUT_WEIGHTS, 'w') as f:
        json.dump(weights, f)
    
    size_mb = os.path.getsize(OUTPUT_WEIGHTS) / (1024 * 1024)
    print(f"\n✅ Pesos exportados: {OUTPUT_WEIGHTS} ({size_mb:.2f} MB)")
    print(f"\nPróximo paso: sube {OUTPUT_WEIGHTS} a HuggingFace")

if __name__ == "__main__":
    main()
