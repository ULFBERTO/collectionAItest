import torch
from model import Transformer
from tokenizer import CharacterTokenizer
import os

# Configuración (Debe coincidir con los hiperparámetros de train.py)
DIM = 384
N_LAYERS = 6
N_HEADS = 12
BLOCK_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoint_transformer.pth"

def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: No se encontró el archivo {CHECKPOINT_PATH}")
        return None, None

    print(f"Cargando checkpoint desde {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Reconstruir Tokenizador
    tokenizer = CharacterTokenizer()
    tokenizer.chars = checkpoint['tokenizer_chars']
    tokenizer.vocab_size = checkpoint['vocab_size']
    tokenizer.stoi = { ch:i for i,ch in enumerate(tokenizer.chars) }
    tokenizer.itos = { i:ch for i,ch in enumerate(tokenizer.chars) }
    
    # Reconstruir Modelo
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=BLOCK_SIZE
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Modo evaluación
    
    print(f"Modelo cargado (Iteración: {checkpoint['iter']}, Loss: {checkpoint['loss']:.4f})")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8):
    """
    Genera una respuesta basada en un prompt.
    Temperature: >1.0 para más creatividad, <1.0 para más precisión.
    """
    # Formatear el prompt con el token especial de usuario
    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    idx = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long).to(DEVICE)
    
    generated_text = ""
    
    print("\nGenerando respuesta...", end="", flush=True)
    
    for _ in range(max_new_tokens):
        # Recortar al tamaño de bloque
        idx_cond = idx[:, -BLOCK_SIZE:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Tomar el último paso, aplicar temperatura y softmax
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Sorteamos el siguiente token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Añadir al tensor y decodificar el nuevo caracter
        idx = torch.cat((idx, idx_next), dim=1)
        char = tokenizer.decode([idx_next.item()])
        
        # Si llegamos al token de fin o el modelo genera su propio marcador, paramos
        if char == "<|end|>" or "<|user|>" in generated_text:
            break
            
        print(char, end="", flush=True)
        generated_text += char
        
    print("\n")

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    if model:
        print("\n--- Modo Chat Iniciado ---")
        print("Escribe 'salir' para terminar.")
        
        while True:
            user_input = input("Tú: ")
            if user_input.lower() == 'salir':
                break
            
            generate_response(model, tokenizer, user_input)
