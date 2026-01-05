import torch
from model import TransformerKiller
from tokenizer import CharacterTokenizer

# Configuración (debe coincidir con train.py)
DIM = 128
STATE_DIM = 16
N_LAYERS = 4
DEVICE = "cpu"  # Forzar CPU para no interferir con el entrenamiento

def load_model():
    checkpoint_path = "ssm_checkpoint.pth"
    
    print("Cargando modelo en CPU...")
    cp = torch.load(checkpoint_path, map_location=DEVICE)
    
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
    ).to(DEVICE)
    
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: Transformer Killer (SSM)")
    print(f"Parámetros: {n_params:,}")
    print(f"Checkpoint: iter {cp.get('iter', '?')}")
    print(f"Vocabulario: {tokenizer.vocab_size} tokens")
    
    return model, tokenizer

def generate(model, tokenizer, prompt, max_tokens=150, temperature=0.8):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(idx)
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Parar si genera token de fin
            if tokenizer.itos.get(idx_next.item(), "") == "<|end|>":
                break
    
    return tokenizer.decode(idx[0].tolist())

def main():
    model, tokenizer = load_model()
    
    print("\n" + "="*50)
    print("  Transformer Killer - Chat (CPU)")
    print("  Escribe 'salir' para terminar")
    print("  Escribe 'reload' para recargar el modelo")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("Tú: ").strip()
            
            if prompt.lower() == "salir":
                print("¡Hasta luego!")
                break
            
            if prompt.lower() == "reload":
                model, tokenizer = load_model()
                print("Modelo recargado.\n")
                continue
            
            if not prompt:
                continue
            
            response = generate(model, tokenizer, prompt)
            print(f"SSM: {response}\n")
            
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
