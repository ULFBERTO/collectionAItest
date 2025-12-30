import torch  # Biblioteca principal de tensores y computación numérica (similar a NumPy pero para IA)
import torch.nn as nn  # Contiene los bloques de construcción de redes neuronales (capas, funciones de pérdida, etc.)
import torch.nn.functional as F  # Funciones matemáticas sin estado (como softmax, silu, etc.) que se usan dentro de las capas
from typing import Optional, Tuple  # Herramientas para indicar tipos de datos, mejorando la legibilidad y detección de errores

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Reemplaza LayerNorm tradicional. No usa 'bias' y no resta la media, 
    solo escala por la raíz cuadrada de la media de los cuadrados.
    Es más eficiente y estadísticamente estable para modelos grandes.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # El único parámetro aprendible es el peso (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x.pow(2).mean(-1, keepdim=True) -> Calcula la media de los cuadrados
        # torch.rsqrt(...) -> 1 / sqrt(valor + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """
    Precomputa las frecuencias para RoPE (Rotary Positional Embeddings).
    Esto es una transformación matemática que 'rota' los vectores de atención
    en el espacio complejo basándose en su posición.
    """
    # head_dim debe ser par para manejar partes reales e imaginarias
    assert head_dim % 2 == 0
    
    # Calculamos los ángulos (theta)
    # theta_i = 10000^(-2i/dim)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta_freq = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
    # Generamos posiciones (0, 1, 2, ..., seq_len-1)
    m = torch.arange(seq_len, device=device)
    
    # Producto exterior para obtener los ángulos finales para cada posición m y dimensión i
    freqs = torch.outer(m, theta_freq).float()
    
    # Convertimos a coordenadas polares (forma compleja: cos + i*sin)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    """
    Aplica la rotación a los tensores x (Query o Key).
    """
    # x tiene forma [Batch, SeqLen, Heads, HeadDim]
    # Convertimos las últimas dimensiones a complejo (fusionando pares)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Reshape freqs para que coincida con el broadcast [1, SeqLen, 1, HeadDim/2]
    # x_complex: [B, S, H, D/2]
    # freqs_complex: [S, D/2] -> [1, S, 1, D/2]
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # Aplicamos la rotación por multiplicación compleja
    x_rotated = x_complex * freqs_complex
    
    # Devolvemos a forma real [Batch, SeqLen, Heads, HeadDim]
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)

class SelfAttention(nn.Module):
    """
    Mecanismo de Auto-Atención Multi-Cabezal con soporte para RoPE.
    """
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Proyecciones para Query, Key y Value
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Proyectamos Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Reshape para multi-cabezal: [Batch, SeqLen, Heads, HeadDim]
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Aplicamos RoPE a Queries y Keys
        xq = apply_rotary_embeddings(xq, freqs_complex)
        xk = apply_rotary_embeddings(xk, freqs_complex)
        
        # Transponemos para computar atención: [Batch, Heads, SeqLen, HeadDim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Scaled Dot-Product Attention: (Q @ K.T) / sqrt(dk)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores + mask # La máscara suele tener -inf en posiciones prohibidas
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # Multiplicamos por V: [Batch, Heads, SeqLen, HeadDim]
        output = torch.matmul(scores, xv)
        
        # Concatenamos cabezales y proyectamos salida
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class SwiGLU(nn.Module):
    """
    Feed-Forward Network usando la activación SwiGLU.
    Es el estándar en modelos modernos como Llama. 
    Usa tres proyecciones: W1, W2 y W3.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU(x) = (Silu(W1(x)) * W3(x)) @ W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    """
    Un bloque completo del Transformer que combina Atención y Feed-Forward.
    Utiliza Pre-Normalization (RMSNorm antes de cada capa) y conexiones residuales.
    """
    def __init__(self, dim: int, n_heads: int, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.attention = SelfAttention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, eps=eps)
        self.ffn_norm = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Sub-capa 1: Atención + Conexión Residual
        # x = x + Attention(RMSNorm(x))
        h = x + self.attention(self.attention_norm(x), freqs_complex, mask)
        
        # Sub-capa 2: Feed-Forward + Conexión Residual
        # out = h + FeedForward(RMSNorm(h))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    """
    La arquitectura completa del modelo GPT desde cero (Decoder-only).
    """
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # 1. Embedding de tokens
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # 2. Bloques de Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, dim * 4) for _ in range(n_layers)
        ])
        
        # 3. Normalización final y salida lineal
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # 4. Precomputar frecuencias de RoPE
        self.register_buffer(
            "freqs_complex", 
            precompute_theta_pos_frequencies(dim // n_heads, max_seq_len, "cpu")
        )

    def forward(self, tokens: torch.Tensor):
        _batch_size, seq_len = tokens.shape
        x = self.tok_embeddings(tokens)
        
        # Extraer frecuencias para la longitud de secuencia actual
        # Aseguramos que freqs_complex esté en el mismo dispositivo que x
        freqs_complex = self.freqs_complex[:seq_len].to(x.device)
        
        # Crear máscara causal (Look-ahead mask)
        # Una matriz triangular superior llena de -inf
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
        
        # Pasar por cada capa del Transformer
        for layer in self.layers:
            x = layer(x, freqs_complex, mask)
            
        x = self.norm(x)
        logits = self.output(x)
        
        return logits

if __name__ == "__main__":
    # Test del modelo completo
    vocab_size = 1000
    dim = 512
    n_layers = 4
    n_heads = 8
    max_seq_len = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Crear modelo
    model = Transformer(vocab_size, dim, n_layers, n_heads, max_seq_len).to(device)
    
    # Simular entrada de tokens (por ejemplo, IDs de palabras)
    tokens = torch.randint(0, vocab_size, (2, 50)).to(device)
    
    # Inferencia
    logits = model(tokens)
    
    print(f"Input shape: {tokens.shape}")
    print(f"Logits shape (Batch, Seq, Vocab): {logits.shape}")
    print("Arquitectura Transformer ensamblada con éxito.")
