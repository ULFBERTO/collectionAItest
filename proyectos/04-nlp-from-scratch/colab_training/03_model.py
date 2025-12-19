# ============================================================
# SECCIÓN 3: MODELO MEJORADO - OxideLLM v2
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class PositionalEncoding(layers.Layer):
    """Positional encoding sinusoidal (más eficiente que embeddings aprendidos)."""
    
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Precomputar encodings
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len]
    
    def get_config(self):
        return {"max_len": self.max_len, "d_model": self.d_model}


class TransformerBlock(layers.Layer):
    """Bloque Transformer con Pre-LayerNorm (más estable)."""
    
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout

        # Pre-LN: LayerNorm antes de atención
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
    
    def call(self, x, training=False):
        # Pre-LN + Attention + Residual
        attn_input = self.ln1(x)
        attn_output = self.mha(
            query=attn_input,
            value=attn_input,
            key=attn_input,
            use_causal_mask=True,
            training=training
        )
        x = x + attn_output
        
        # Pre-LN + FFN + Residual
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        x = x + ffn_output
        
        return x
    
    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate
        }


class OxideLLM(Model):
    """
    OxideLLM v2 - Modelo de lenguaje mejorado.
    
    Configuraciones sugeridas:
    - Small (~25M):  d_model=512,  num_heads=8,  dff=2048, num_layers=6
    - Medium (~50M): d_model=768,  num_heads=12, dff=3072, num_layers=8
    - Large (~100M): d_model=1024, num_heads=16, dff=4096, num_layers=12
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        dff=2048,
        num_layers=6,
        max_len=512,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout
        
        # Token embeddings
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.dropout = layers.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout, name=f"block_{i}")
            for i in range(num_layers)
        ]
        
        # Final layer norm (Pre-LN style)
        self.final_ln = layers.LayerNormalization(epsilon=1e-6)
        
        # Output projection
        self.output_layer = layers.Dense(vocab_size)
    
    def call(self, x, training=False):
        # Embedding + Positional
        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Final norm + projection
        x = self.final_ln(x)
        logits = self.output_layer(x)
        
        return logits
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "num_layers": self.num_layers,
            "max_len": self.max_len,
            "dropout": self.dropout_rate
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def count_params(self):
        """Cuenta parámetros del modelo."""
        return sum(p.numpy().size for p in self.trainable_weights)


def create_model(vocab_size, size="medium"):
    """Factory para crear modelos de diferentes tamaños."""
    configs = {
        "small": {"d_model": 512, "num_heads": 8, "dff": 2048, "num_layers": 6},
        "medium": {"d_model": 768, "num_heads": 12, "dff": 3072, "num_layers": 8},
        "large": {"d_model": 1024, "num_heads": 16, "dff": 4096, "num_layers": 12},
    }
    
    config = configs.get(size, configs["medium"])
    model = OxideLLM(vocab_size=vocab_size, **config)
    
    # Build model
    dummy = tf.zeros((1, 512), dtype=tf.int32)
    model(dummy)
    
    params = model.count_params()
    print(f"✅ Modelo '{size}' creado: {params:,} parámetros ({params/1e6:.1f}M)")
    
    return model
