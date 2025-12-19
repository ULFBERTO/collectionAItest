"""
OxideLearn v2 - Modelo Base Mejorado (~125M params)
Arquitectura Transformer decoder-only optimizada.
"""

import tensorflow as tf
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    """Codificación posicional sinusoidal."""
    
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Precalcular encodings
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
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class TransformerBlock(tf.keras.layers.Layer):
    """Bloque Transformer con Pre-LN (más estable para modelos grandes)."""
    
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='gelu'),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(self.d_model),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])
        super().build(input_shape)
    
    def call(self, x, training=False):
        # Pre-LN Self-Attention
        attn_input = self.ln1(x)
        attn_output = self.mha(
            query=attn_input,
            value=attn_input,
            key=attn_input,
            use_causal_mask=True,
            training=training
        )
        x = x + attn_output
        
        # Pre-LN FFN
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        return x + ffn_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate
        })
        return config


class OxideModel(tf.keras.Model):
    """
    OxideLearn v2 - Modelo principal.
    
    Arquitectura GPT-like con ~125M parámetros.
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        dff=3072,
        max_seq_len=512,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        
        # Capas
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        self.blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout, name=f"block_{i}")
            for i in range(num_layers)
        ]
        
        self.final_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=False):
        # Embedding + Positional
        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Output
        x = self.final_ln(x)
        logits = self.output_layer(x)
        
        return logits
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dff": self.dff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout_rate
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_model(config: dict) -> OxideModel:
    """Crea una instancia del modelo."""
    model = OxideModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dff=config["dff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"]
    )
    
    # Build model
    dummy = tf.zeros((1, 32), dtype=tf.int32)
    model(dummy)
    
    return model


def count_parameters(model: tf.keras.Model) -> int:
    """Cuenta parámetros entrenables."""
    return sum(np.prod(w.shape) for w in model.trainable_weights)


if __name__ == "__main__":
    from config import MODEL_CONFIG
    
    print("Creando modelo OxideLearn v2...")
    model = create_model(MODEL_CONFIG)
    params = count_parameters(model)
    
    print(f"\n📊 Modelo creado:")
    print(f"   Parámetros: {params:,} ({params/1e6:.1f}M)")
    print(f"   Vocab size: {MODEL_CONFIG['vocab_size']}")
    print(f"   d_model: {MODEL_CONFIG['d_model']}")
    print(f"   Capas: {MODEL_CONFIG['num_layers']}")
    print(f"   Cabezas: {MODEL_CONFIG['num_heads']}")
    
    # Test forward pass
    test_input = tf.random.uniform((2, 64), 0, 1000, dtype=tf.int32)
    output = model(test_input)
    print(f"\n   Test input: {test_input.shape}")
    print(f"   Test output: {output.shape}")
