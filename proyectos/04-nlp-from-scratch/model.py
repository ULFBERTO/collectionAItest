import tensorflow as tf
from tensorflow.keras import layers, models

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len=2048, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_embedding = layers.Embedding(max_len, d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_embedding(positions)
        x = self.embedding(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return config

class CausalSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.dropout = layers.Dropout(dropout)
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization()

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        attn_output = self.dropout(attn_output)
        return self.add([x, attn_output])

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
        })
        return config

class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout
        self.seq = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization()

    def call(self, x):
        ff_output = self.seq(x)
        return self.add([x, ff_output])

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dff": self.dff,
            "dropout": self.dropout_rate,
        })
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, x):
        x = self.layernorm1(self.attn(x))
        x = self.layernorm2(self.ffn(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate,
        })
        return config

class GPTModel(models.Model):
    def __init__(self, vocab_size, d_model=256, num_heads=4, dff=512, num_layers=4, max_len=100, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model, max_len)
        self.blocks = [TransformerBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x):
        x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)
        logits = self.final_layer(x)
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "num_layers": self.num_layers,
            "max_len": self.max_len,
            "dropout": self.dropout_rate,
        })
        return config

if __name__ == "__main__":
    # Test simple
    vocab_size = 100
    model = GPTModel(vocab_size=vocab_size, d_model=64, num_heads=2, dff=128, num_layers=2)
    dummy_input = tf.random.uniform((1, 50), minval=0, maxval=vocab_size, dtype=tf.int32)
    output = model(dummy_input)
    print("Model Output Shape:", output.shape) # (1, 50, 100)
