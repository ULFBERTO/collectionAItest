import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class SSMBlock(nn.Module):
    """
    Una implementación simplificada de una capa de State Space Model (SSM) Selectivo.
    A diferencia del Transformer, esta capa tiene memoria LINEAL.
    """
    def __init__(self, dim: int, state_dim: int = 16, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.inner_dim = expand * dim
        
        # Proyecciones de entrada
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        
        # Convolución 1D para capturar contexto local (estilo Mamba)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=4,
            groups=self.inner_dim,
            padding=3
        )
        
        # Parámetros del SSM (Simplificado)
        # Delta (dt): El paso de tiempo reactivo
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=True)
        
        # matrices A y B (Simplificadas para ser selectivas)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float().repeat(self.inner_dim, 1)))
        self.B_proj = nn.Linear(self.inner_dim, state_dim, bias=False)
        self.C_proj = nn.Linear(self.inner_dim, state_dim, bias=False)
        
        # Proyección de salida
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [Batch, SeqLen, Dim]
        b, l, d = x.shape
        
        # 1. Proyección inicial y split
        x_and_res = self.in_proj(x) # [B, L, 2 * InnerDim]
        x_inner, res = x_and_res.split(self.inner_dim, dim=-1)
        
        # 2. Convolución local
        x_inner = x_inner.transpose(1, 2) # [B, InnerDim, L]
        x_inner = self.conv1d(x_inner)[:, :, :l] # Recortar padding
        x_inner = x_inner.transpose(1, 2) # [B, L, InnerDim]
        x_inner = F.silu(x_inner) # Activación Swish
        
        # 3. Mecanismo SSM Selectivo (Simplificado)
        # En una implementación real (como Mamba), esto se hace con un kernel de GPU 
        # para ser ultra rápido. Aquí usamos un bucle o aproximación para entenderlo.
        
        dt = F.softplus(self.dt_proj(x_inner)) # [B, L, InnerDim]
        A = -torch.exp(self.A_log) # [InnerDim, StateDim]
        B = self.B_proj(x_inner) # [B, L, StateDim]
        C = self.C_proj(x_inner) # [B, L, StateDim]
        
        # El "Estado Oculto" del modelo (Memory)
        # Aquí es donde ocurre la magia: el estado tiene tamaño fijo [B, InnerDim, StateDim]
        state = torch.zeros(b, self.inner_dim, self.state_dim, device=x.device)
        y = torch.zeros(b, l, self.inner_dim, device=x.device)
        
        # Escaneo Secuencial (Selective Scan)
        # Esto reemplaza a la Atención del Transformer.
        for t in range(l):
            # dt_t: [B, InnerDim]
            dt_t = dt[:, t, :].unsqueeze(-1)
            # x_t: [B, InnerDim]
            x_t = x_inner[:, t, :].unsqueeze(-1)
            # B_t: [B, StateDim]
            B_t = B[:, t, :].unsqueeze(1)
            
            # Discretización (Aproximación de Euler)
            A_bar = torch.exp(A.unsqueeze(0) * dt_t) # [B, InnerDim, StateDim]
            B_bar = dt_t * B_t
            
            # Actualizar Estado: h = A*h + B*x
            state = A_bar * state + B_bar * x_t
            
            # Salida: y = C*h
            C_t = C[:, t, :].unsqueeze(-1) # [B, StateDim, 1]
            y[:, t, :] = torch.matmul(state, C_t).squeeze(-1)
        
        # 4. Combinar con el residuo y proyectar salida
        out = y * F.silu(res)
        return self.out_proj(out)

class TransformerKiller(nn.Module):
    """
    Arquitectura basada en SSM (Mamba-like).
    No usa Atención. Contexto teóricamente infinito.
    """
    def __init__(self, vocab_size: int, dim: int, n_layers: int, state_dim: int = 16):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': RMSNorm(dim),
                'ssm': SSMBlock(dim, state_dim=state_dim)
            }) for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        x = self.tok_embeddings(tokens)
        
        for layer in self.layers:
            # Conexión residual + SSM
            x = x + layer['ssm'](layer['norm'](x))
            
        x = self.norm_f(x)
        return self.output(x)

if __name__ == "__main__":
    # Test de dimensiones
    model = TransformerKiller(vocab_size=100, dim=128, n_layers=4)
    test_input = torch.randint(0, 100, (2, 50))
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"SSM Output shape: {output.shape}")
    # Nota como no hay límite de BLOCK_SIZE en el forward, 
    # solo el que dicte tu RAM, pero el coste es Lineal!
