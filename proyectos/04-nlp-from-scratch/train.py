import tensorflow as tf
import os
import json
import time
import numpy as np

# ============================================================
# CONFIGURACIÓN - Similar a Colab
# ============================================================

# Modelo
MODEL_SIZE = "small"  # "small", "medium", "large"
CONFIGS = {
    "small": {"d_model": 512, "num_heads": 8, "dff": 2048, "num_layers": 6},      # ~25M params
    "medium": {"d_model": 768, "num_heads": 12, "dff": 3072, "num_layers": 8},    # ~50M params
    "large": {"d_model": 1024, "num_heads": 16, "dff": 4096, "num_layers": 12},   # ~100M params
}

# Entrenamiento
EPOCHS = 20
BATCH_SIZE = 32
SEQ_LENGTH = 512
LEARNING_RATE = 1e-4
DROPOUT = 0.1
SAVE_EVERY = 5

# Tokenizer BPE
VOCAB_SIZE = 8000

# Rutas
CORPUS_PATH = "../Data/libros_espanol/corpus_completo.txt"
CHECKPOINT_DIR = './training_checkpoints'
MODEL_SAVE_DIR = './OxideLLM_5M_saved_model'
TOKENIZER_DIR = './tokenizer'

# Hugging Face
HF_REPO_ID = "ULFBERTO/OxideLLM_5M"
UPLOAD_TO_HF = False  # Cambiar a True para subir automáticamente


# ============================================================
# MODELO
# ============================================================

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        return x + self.pe[:tf.shape(x)[1]]
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='gelu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout)
        ])
    
    def call(self, x, training=False):
        attn_input = self.ln1(x)
        attn_output = self.mha(
            query=attn_input, value=attn_input, key=attn_input, 
            use_causal_mask=True, training=training
        )
        x = x + attn_output
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        return x + ffn_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dff": self.dff, "dropout": self.dropout_rate
        })
        return config


class OxideLLM(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, num_heads=8, dff=2048, 
                 num_layers=6, max_len=512, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout, name=f"block_{i}") 
            for i in range(num_layers)
        ]
        self.final_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x, training=False):
        x = self.embedding(x)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout_layer(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.final_ln(x)
        return self.output_layer(x)
    
    def get_config(self):
        return {
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "num_heads": self.num_heads, "dff": self.dff,
            "num_layers": self.num_layers, "max_len": self.max_len,
            "dropout": self.dropout_rate
        }


# ============================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================

def load_corpus():
    """Carga el corpus de texto."""
    if os.path.exists(CORPUS_PATH):
        print(f"📚 Cargando corpus: {CORPUS_PATH}")
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Fallback: Don Quijote
        don_quijote_path = "don_quijote.txt"
        if not os.path.exists(don_quijote_path):
            import requests
            url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
            print(f"📥 Descargando Don Quijote...")
            response = requests.get(url)
            text = response.text
            with open(don_quijote_path, 'w', encoding='utf-8') as f:
                f.write(text)
        else:
            with open(don_quijote_path, 'r', encoding='utf-8') as f:
                text = f.read()
    
    print(f"✅ Corpus cargado: {len(text):,} caracteres")
    return text


def train_tokenizer(corpus):
    """Entrena tokenizer SentencePiece BPE."""
    import sentencepiece as spm
    
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    model_prefix = os.path.join(TOKENIZER_DIR, "oxide_bpe")
    
    # Si ya existe, cargar
    if os.path.exists(f"{model_prefix}.model"):
        print(f"📖 Cargando tokenizer existente...")
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        return sp
    
    # Entrenar nuevo
    print(f"🔧 Entrenando tokenizer BPE (vocab_size={VOCAB_SIZE})...")
    temp_file = os.path.join(TOKENIZER_DIR, "temp_corpus.txt")
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(corpus)
    
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
    )
    
    os.remove(temp_file)
    
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    print(f"✅ Tokenizer entrenado: {sp.get_piece_size()} tokens")
    
    return sp


def create_dataset(tokens):
    """Crea dataset de TensorFlow."""
    tokens = np.array(tokens, dtype=np.int32)
    print(f"📊 Tokens totales: {len(tokens):,}")
    
    def make_sequences():
        for i in range(0, len(tokens) - SEQ_LENGTH, SEQ_LENGTH // 2):
            chunk = tokens[i:i + SEQ_LENGTH + 1]
            if len(chunk) == SEQ_LENGTH + 1:
                yield chunk[:-1], chunk[1:]
    
    dataset = tf.data.Dataset.from_generator(
        make_sequences,
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LENGTH,), dtype=tf.int32),
            tf.TensorSpec(shape=(SEQ_LENGTH,), dtype=tf.int32)
        )
    )
    
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def train():
    """Función principal de entrenamiento."""
    print("\n" + "=" * 60)
    print("🧠 OxideLLM - Entrenamiento Local")
    print("=" * 60)
    
    # 1. Cargar corpus
    corpus = load_corpus()
    
    # 2. Entrenar/cargar tokenizer
    sp = train_tokenizer(corpus)
    actual_vocab_size = sp.get_piece_size()
    
    # 3. Tokenizar corpus
    print("🔄 Tokenizando corpus...")
    all_tokens = sp.encode(corpus)
    
    # 4. Crear dataset
    dataset = create_dataset(all_tokens)
    
    # 5. Crear modelo
    config = CONFIGS[MODEL_SIZE]
    print(f"\n📐 Configuración del modelo ({MODEL_SIZE}):")
    print(f"   d_model: {config['d_model']}")
    print(f"   num_heads: {config['num_heads']}")
    print(f"   dff: {config['dff']}")
    print(f"   num_layers: {config['num_layers']}")
    print(f"   vocab_size: {actual_vocab_size}")
    print(f"   seq_length: {SEQ_LENGTH}")
    
    model = OxideLLM(
        vocab_size=actual_vocab_size,
        max_len=SEQ_LENGTH,
        dropout=DROPOUT,
        **config
    )
    
    # Build model
    dummy = tf.zeros((1, SEQ_LENGTH), dtype=tf.int32)
    model(dummy)
    
    total_params = sum(p.numpy().size for p in model.trainable_weights)
    print(f"   Parámetros: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 6. Configurar entrenamiento
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    history = {"loss": [], "accuracy": [], "epoch_time": []}
    
    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = tf.reduce_mean(loss_fn(targets, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss.update_state(loss)
        train_accuracy.update_state(targets, predictions)
        return loss
    
    # 7. Entrenar
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n🚀 Iniciando entrenamiento por {EPOCHS} épocas")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        for batch, (inputs, targets) in enumerate(dataset):
            loss = train_step(inputs, targets)
            if batch % 50 == 0:
                print(f"  Batch {batch}: loss={loss:.4f}", end="\r")
        
        epoch_time = time.time() - start_time
        history["loss"].append(float(train_loss.result()))
        history["accuracy"].append(float(train_accuracy.result()))
        history["epoch_time"].append(epoch_time)
        
        print(f"Época {epoch+1}/{EPOCHS} | Loss: {train_loss.result():.4f} | "
              f"Acc: {train_accuracy.result():.4f} | Time: {epoch_time:.1f}s")
        
        # Guardar checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch+1}.weights.h5")
            model.save_weights(ckpt_path)
            print(f"  💾 Checkpoint: {ckpt_path}")
    
    print("\n✅ Entrenamiento completado")
    
    # 8. Guardar modelo final
    save_model(model, sp, config, actual_vocab_size, history)
    
    return model, sp, history


def save_model(model, sp, config, vocab_size, history):
    """Guarda el modelo y archivos asociados."""
    import shutil
    
    print(f"\n💾 Guardando modelo en {MODEL_SAVE_DIR}...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(MODEL_SAVE_DIR, "saved_model")
    model.save(model_path)
    
    # Guardar pesos H5
    model.save_weights(os.path.join(MODEL_SAVE_DIR, "model_weights.h5"))
    
    # Copiar tokenizer
    tokenizer_src = os.path.join(TOKENIZER_DIR, "oxide_bpe.model")
    tokenizer_vocab = os.path.join(TOKENIZER_DIR, "oxide_bpe.vocab")
    if os.path.exists(tokenizer_src):
        shutil.copy(tokenizer_src, os.path.join(MODEL_SAVE_DIR, "tokenizer.model"))
    if os.path.exists(tokenizer_vocab):
        shutil.copy(tokenizer_vocab, os.path.join(MODEL_SAVE_DIR, "tokenizer.vocab"))
    
    # Crear vocab.json para compatibilidad
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
    with open(os.path.join(MODEL_SAVE_DIR, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Guardar configuración
    model_config = {
        "vocab_size": vocab_size,
        "d_model": config["d_model"],
        "num_heads": config["num_heads"],
        "dff": config["dff"],
        "num_layers": config["num_layers"],
        "max_len": SEQ_LENGTH,
        "dropout": DROPOUT,
        "model_type": "oxide_llm"
    }
    with open(os.path.join(MODEL_SAVE_DIR, "config.json"), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Guardar historial
    with open(os.path.join(MODEL_SAVE_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("✅ Modelo guardado")


def generate_text(model, sp, prompt, max_tokens=100, temperature=0.8):
    """Genera texto con el modelo."""
    tokens = sp.encode(prompt)
    
    for _ in range(max_tokens):
        input_ids = tf.constant([tokens[-SEQ_LENGTH:]], dtype=tf.int32)
        logits = model(input_ids, training=False)
        logits = logits[0, -1, :] / temperature
        
        # Top-k sampling
        top_k = 40
        top_logits, top_indices = tf.math.top_k(logits, k=top_k)
        probs = tf.nn.softmax(top_logits)
        next_idx = tf.random.categorical(tf.expand_dims(tf.math.log(probs), 0), 1)
        next_token = top_indices[next_idx[0, 0]].numpy()
        
        tokens.append(int(next_token))
        if next_token == 3:  # </s>
            break
    
    return sp.decode(tokens)


def upload_to_huggingface(repo_id=None, token=None):
    """Sube el modelo a Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, login, create_repo
        
        repo_id = repo_id or HF_REPO_ID
        
        if token:
            login(token=token)
        
        api = HfApi()
        
        try:
            create_repo(repo_id, exist_ok=True)
        except:
            pass
        
        print(f"📤 Subiendo a {repo_id}...")
        api.upload_folder(folder_path=MODEL_SAVE_DIR, repo_id=repo_id, repo_type="model")
        print(f"✅ Subido: https://huggingface.co/{repo_id}")
        
    except ImportError:
        print("❌ huggingface_hub no instalado. Ejecuta: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ Error al subir: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar OxideLLM")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Tamaño del batch")
    parser.add_argument("--model-size", type=str, default=MODEL_SIZE, choices=["small", "medium", "large"])
    parser.add_argument("--upload", action="store_true", help="Subir a HF después de entrenar")
    parser.add_argument("--repo", type=str, default=None, help="Repo ID de Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Token de Hugging Face")
    parser.add_argument("--test", action="store_true", help="Probar generación después de entrenar")
    args = parser.parse_args()
    
    # Actualizar configuración
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MODEL_SIZE = args.model_size
    
    # Entrenar
    model, sp, history = train()
    
    # Probar generación
    if args.test:
        print("\n🧪 Probando generación:\n")
        prompts = ["En un lugar de la Mancha", "El amor es", "La vida es como"]
        for prompt in prompts:
            print(f"Prompt: '{prompt}'")
            result = generate_text(model, sp, prompt, max_tokens=50)
            print(f"Output: {result}\n")
    
    # Subir a Hugging Face
    if args.upload or UPLOAD_TO_HF:
        print("\n🤗 Subiendo modelo a Hugging Face Hub...")
        upload_to_huggingface(repo_id=args.repo, token=args.token)
