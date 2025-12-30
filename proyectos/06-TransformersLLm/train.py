import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Transformer
from tokenizer import CharacterTokenizer, load_books
import time

# Configuración de Hiperparámetros (Optimizado para ~10 Horas en CPU)
BATCH_SIZE = 32          # Lote equilibrado para CPU
BLOCK_SIZE = 256         # Aumentado: el modelo recordará párrafos más largos (antes 128)
DIM = 384                # Aumentado: modelo con más "capacidad intelectual" (antes 256)
N_LAYERS = 6             # Profundidad mantenida para estabilidad
N_HEADS = 12             # Aumentado: debe ser divisor de DIM (384/12 = 32 dim por cabezal)
LEARNING_RATE = 3e-4     # Tasa de aprendizaje estándar para estabilidad
MAX_ITERS = 20000        # Aumentado significativamente para llenar las 10 horas
EVAL_INTERVAL = 500      # Evaluación cada 500 pasos
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataset(Dataset):
    """
    Dataset de PyTorch para manejar el texto de los libros.
    Crea fragmentos de longitud BLOCK_SIZE.
    """
    def __init__(self, data_ids, block_size):
        self.data = torch.tensor(data_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x es la secuencia de entrada
        # y es la secuencia objetivo (la misma que x pero desplazada un paso a la derecha)
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

def get_batch(data_ids, batch_size, block_size):
    """
    Función auxiliar para obtener un lote aleatorio (alternativa a DataLoader para simplicidad).
    """
    ix = torch.randint(len(data_ids) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data_ids[i : i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(data_ids[i + 1 : i + block_size + 1]) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters=100):
    """
    Estima la pérdida en los sets de entrenamiento y validación sin afectar el entrenamiento.
    """
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = Y.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate(model, tokenizer, max_new_tokens=100, prompt=""):
    """
    Genera texto nuevo a partir de un prompt opcional.
    """
    model.eval()
    # Si no hay prompt, empezamos con un token aleatorio o espacio
    idx = torch.tensor([tokenizer.encode(prompt if prompt else " ")], dtype=torch.long).to(DEVICE)
    
    for _ in range(max_new_tokens):
        # Recortamos al tamaño de bloque si es necesario
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)
        # Tomamos solo el último paso de tiempo
        logits = logits[:, -1, :]
        # Aplicamos softmax para obtener probabilidades
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Sorteamos el siguiente token
        idx_next = torch.multinomial(probs, num_samples=1)
        # Lo añadimos a la secuencia
        idx = torch.cat((idx, idx_next), dim=1)
    
    model.train()
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    # 1. Cargar datos
    print(f"Usando dispositivo: {DEVICE}")
    books_path = r"C:\EVIROMENT\M\collectionAItest\proyectos\Data\libros_espanol"
    text = load_books(books_path)
    
    # 2. Inicializar tokenizador y preparar IDs
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    data_ids = tokenizer.encode(text)
    
    # Separar en entrenamiento (90%) y validación (10%)
    n = int(0.9 * len(data_ids))
    train_data = data_ids[:n]
    val_data = data_ids[n:]
    
    # 3. Inicializar modelo y optimizador
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        dim=DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=BLOCK_SIZE
    ).to(DEVICE)
    
    # Calcular y mostrar el número exacto de parámetros
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Número de parámetros del modelo: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Iniciando entrenamiento ({MAX_ITERS} iteraciones)...")
    start_time = time.time()
    
    for iter in range(MAX_ITERS):
        # Obtener lote y calcular pérdida
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        
        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = yb.view(B*T)
        
        loss = nn.functional.cross_entropy(logits, targets)
        
        # Retropropagación
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Evaluación periódica y Guardado de Checkpoints
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, BATCH_SIZE, BLOCK_SIZE)
            elapsed = time.time() - start_time
            print(f"Iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, tiempo: {elapsed:.2f}s")
            
            # Generar una muestra para ver progreso
            sample = generate(model, tokenizer, max_new_tokens=100)
            print(f"Muestra generada:\n--- {sample}\n---")

            # GUARDAR CHECKPOINT (Por si se corta la luz o el proceso)
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses['val'],
                'vocab_size': tokenizer.vocab_size,
                'tokenizer_chars': tokenizer.chars
            }, "checkpoint_transformer.pth")

    # Guardar el modelo al terminar
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_chars': tokenizer.chars
    }, "transformer_v1.pth")
    print("Modelo guardado como transformer_v1.pth")
