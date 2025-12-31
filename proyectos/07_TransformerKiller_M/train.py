import torch
import torch.nn as nn
from model import TransformerKiller
from tokenizer import CharacterTokenizer, load_books
import time
import os

# Hiperparámetros SSM (Contexto Masivo)
BATCH_SIZE = 8           # Bajamos un poco el batch para compensar el contexto largo en RAM
BLOCK_SIZE = 2048        # ¡Aumentado a 2048! (4 veces más que antes)
DIM = 128
STATE_DIM = 16           # El tamaño de la "memoria" interna de la capa
N_LAYERS = 4
LEARNING_RATE = 1e-3
MAX_ITERS = 5000
EVAL_INTERVAL = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(data_ids, batch_size, block_size):
    ix = torch.randint(len(data_ids) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data_ids[i : i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(data_ids[i + 1 : i + block_size + 1]) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
            logits = model(X)
            B, L, C = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B*L, C), Y.view(B*L))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate(model, tokenizer, max_new_tokens=100, prompt=" "):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(DEVICE)
    for _ in range(max_new_tokens):
        # En SSM NO necesitamos recortar el contexto (no hay BLOCK_SIZE técnico)
        # pero por estabilidad lo dejamos en un valor razonable.
        logits = model(idx)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    print(f"Iniciando Proyecto: Transformer Killer (SSM)")
    print(f"Usando: {DEVICE}")
    
    # 1. Datos
    books_path = r"C:\EVIROMENT\M\collectionAItest\proyectos\Data\libros_espanol"
    text = load_books(books_path)
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    data_ids = tokenizer.encode(text)
    
    n = int(0.9 * len(data_ids))
    train_data = data_ids[:n]
    val_data = data_ids[n:]
    
    # 2. Modelo
    model = TransformerKiller(
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        n_layers=N_LAYERS,
        state_dim=STATE_DIM
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros SSM: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Reanudación
    checkpoint_path = "ssm_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print("Reanudando SSM...")
        cp = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])

    print("Comenzando entrenamiento...")
    for iter in range(MAX_ITERS):
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"Iter {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            print(f"Generado: {generate(model, tokenizer, max_new_tokens=50)}")
            
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer_chars': tokenizer.chars
            }, checkpoint_path)

        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        logits = model(xb)
        B, L, C = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B*L, C), yb.view(B*L))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
