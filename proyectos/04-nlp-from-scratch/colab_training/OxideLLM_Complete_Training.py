# ============================================================
# NOTEBOOK COMPLETO: OxideLLM Training
# ============================================================
# Copiar cada sección en celdas separadas de Google Colab
# Ejecutar en orden de arriba a abajo
# ============================================================

# %% [markdown]
# # 🧠 OxideLLM - Entrenamiento Completo
# 
# Este notebook entrena OxideLLM desde cero y lo sube a HuggingFace.
# 
# **Requisitos:**
# - GPU en Colab (Runtime > Change runtime type > GPU)
# - Datos en Google Drive
# - Token de HuggingFace

# %% [markdown]
# ## 1️⃣ Setup Inicial

# %%
# ⚠️ IMPORTANTE: Activa GPU primero!
# Runtime > Change runtime type > GPU (T4)

# Verificar GPU
!nvidia-smi

# Si dice "command not found", activa GPU arriba y reinicia el runtime

# Instalar dependencias
!pip install -q sentencepiece huggingface_hub tensorflowjs --upgrade

# Verificar TensorFlow
import tensorflow as tf
print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ GPU: {tf.config.list_physical_devices('GPU')}")

# Montar Drive
from google.colab import drive
drive.mount('/content/drive')

import os

# ⚠️ CONFIGURAR ESTAS RUTAS según tu Drive
DATA_PATH = "/content/drive/MyDrive/OxideLLM/libros_espanol"
OUTPUT_PATH = "/content/drive/MyDrive/OxideLLM/checkpoints"
TOKENIZER_PATH = "/content/drive/MyDrive/OxideLLM/tokenizer"

# Tu token de HuggingFace (obtener en https://huggingface.co/settings/tokens)
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # ⚠️ CAMBIAR

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(TOKENIZER_PATH, exist_ok=True)

print("✅ Setup completado")

# %% [markdown]
# ## 2️⃣ Cargar y Preparar Datos

# %%
import glob
import sentencepiece as spm
import tensorflow as tf
import numpy as np

# Cargar todos los textos
all_text = []
txt_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
print(f"📚 Encontrados {len(txt_files)} archivos")

for filepath in sorted(txt_files):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        all_text.append(text)
        print(f"  ✓ {os.path.basename(filepath)}: {len(text):,} chars")

corpus = "\n\n".join(all_text)
print(f"\n📊 Total: {len(corpus):,} caracteres")


# %% [markdown]
# ## 3️⃣ Entrenar Tokenizer BPE

# %%
VOCAB_SIZE = 8000  # Tamaño del vocabulario

# Guardar corpus temporal
temp_file = "/content/temp_corpus.txt"
with open(temp_file, 'w', encoding='utf-8') as f:
    f.write(corpus)

# Entrenar SentencePiece
model_prefix = os.path.join(TOKENIZER_PATH, "oxide_bpe")

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

# Cargar tokenizer
sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")

os.remove(temp_file)
print(f"✅ Tokenizer entrenado: {sp.get_piece_size()} tokens")

# Probar tokenización
test_text = "En un lugar de la Mancha"
tokens = sp.encode(test_text)
decoded = sp.decode(tokens)
print(f"Test: '{test_text}' -> {tokens} -> '{decoded}'")

# %% [markdown]
# ## 4️⃣ Crear Dataset

# %%
SEQ_LENGTH = 512
BATCH_SIZE = 32

# Tokenizar corpus completo
all_tokens = sp.encode(corpus)
all_tokens = np.array(all_tokens, dtype=np.int32)
print(f"📊 Tokens totales: {len(all_tokens):,}")

# Crear secuencias con overlap
def make_sequences():
    for i in range(0, len(all_tokens) - SEQ_LENGTH, SEQ_LENGTH // 2):
        chunk = all_tokens[i:i + SEQ_LENGTH + 1]
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

# Contar batches
num_batches = sum(1 for _ in dataset)
print(f"✅ Dataset creado: {num_batches} batches de {BATCH_SIZE}")

# Recrear dataset (el conteo lo consumió)
dataset = tf.data.Dataset.from_generator(
    make_sequences,
    output_signature=(
        tf.TensorSpec(shape=(SEQ_LENGTH,), dtype=tf.int32),
        tf.TensorSpec(shape=(SEQ_LENGTH,), dtype=tf.int32)
    )
).shuffle(10000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# %% [markdown]
# ## 5️⃣ Definir Modelo

# %%
from tensorflow.keras import layers, Model

class PositionalEncoding(layers.Layer):
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
        return {"max_len": self.max_len, "d_model": self.d_model}


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
    
    def call(self, x, training=False):
        attn_input = self.ln1(x)
        attn_output = self.mha(query=attn_input, value=attn_input, key=attn_input, use_causal_mask=True, training=training)
        x = x + attn_output
        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        return x + ffn_output
    
    def get_config(self):
        return {"d_model": self.d_model, "num_heads": self.num_heads, "dff": self.dff, "dropout": self.dropout_rate}


class OxideLLM(Model):
    def __init__(self, vocab_size, d_model=512, num_heads=8, dff=2048, num_layers=6, max_len=512, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.dropout_layer = layers.Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, num_heads, dff, dropout, name=f"block_{i}") for i in range(num_layers)]
        self.final_ln = layers.LayerNormalization(epsilon=1e-6)
        self.output_layer = layers.Dense(vocab_size)
    
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
        return {"vocab_size": self.vocab_size, "d_model": self.d_model, "num_heads": self.num_heads,
                "dff": self.dff, "num_layers": self.num_layers, "max_len": self.max_len, "dropout": self.dropout_rate}


# %% [markdown]
# ## 6️⃣ Crear y Compilar Modelo

# %%
# ⚠️ CONFIGURACIÓN DEL MODELO - Ajustar según GPU disponible
# "small":  ~25M params - d_model=512,  num_heads=8,  dff=2048, num_layers=6
# "medium": ~50M params - d_model=768,  num_heads=12, dff=3072, num_layers=8
# "large":  ~100M params - d_model=1024, num_heads=16, dff=4096, num_layers=12

MODEL_SIZE = "small"  # Cambiar a "medium" o "large" si tienes GPU potente

configs = {
    "small": {"d_model": 512, "num_heads": 8, "dff": 2048, "num_layers": 6},
    "medium": {"d_model": 768, "num_heads": 12, "dff": 3072, "num_layers": 8},
    "large": {"d_model": 1024, "num_heads": 16, "dff": 4096, "num_layers": 12},
}

config = configs[MODEL_SIZE]
model = OxideLLM(
    vocab_size=VOCAB_SIZE,
    max_len=SEQ_LENGTH,
    **config
)

# Build model
dummy = tf.zeros((1, SEQ_LENGTH), dtype=tf.int32)
model(dummy)

# Contar parámetros
total_params = sum(p.numpy().size for p in model.trainable_weights)
print(f"✅ Modelo '{MODEL_SIZE}' creado: {total_params:,} parámetros ({total_params/1e6:.1f}M)")

# %% [markdown]
# ## 7️⃣ Entrenar

# %%
import time
import json

EPOCHS = 20  # Ajustar según tiempo disponible
LEARNING_RATE = 1e-4
SAVE_EVERY = 5  # Guardar cada N épocas

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
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

print(f"\n🚀 Iniciando entrenamiento por {EPOCHS} épocas")
print("=" * 60)

for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss.reset_state()
    train_accuracy.reset_state()
    
    for batch, (inputs, targets) in enumerate(dataset):
        loss = train_step(inputs, targets)
        if batch % 100 == 0:
            print(f"  Batch {batch}: loss={loss:.4f}", end="\r")
    
    epoch_time = time.time() - start_time
    history["loss"].append(float(train_loss.result()))
    history["accuracy"].append(float(train_accuracy.result()))
    history["epoch_time"].append(epoch_time)
    
    print(f"Época {epoch+1}/{EPOCHS} | Loss: {train_loss.result():.4f} | Acc: {train_accuracy.result():.4f} | Time: {epoch_time:.1f}s")
    
    if (epoch + 1) % SAVE_EVERY == 0:
        ckpt_path = os.path.join(OUTPUT_PATH, f"ckpt_epoch_{epoch+1}")
        model.save_weights(ckpt_path)
        print(f"  💾 Checkpoint: {ckpt_path}")

print("\n✅ Entrenamiento completado")

# Guardar modelo final
final_path = os.path.join(OUTPUT_PATH, "oxide_llm_final")
model.save(final_path)
print(f"💾 Modelo guardado: {final_path}")

# Guardar historial
with open(os.path.join(OUTPUT_PATH, "history.json"), 'w') as f:
    json.dump(history, f)


# %% [markdown]
# ## 8️⃣ Probar Generación

# %%
def generate_text(prompt, max_tokens=100, temperature=0.8):
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

# Probar
print("🧪 Probando generación:\n")
prompts = [
    "En un lugar de la Mancha",
    "El amor es",
    "La vida es como",
]

for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    result = generate_text(prompt, max_tokens=50)
    print(f"Output: {result}\n")
    print("-" * 50)

# %% [markdown]
# ## 9️⃣ Subir a HuggingFace

# %%
from huggingface_hub import HfApi, login, create_repo
import shutil

# Login
login(token=HF_TOKEN)
api = HfApi()

# Preparar exportación
export_dir = os.path.join(OUTPUT_PATH, "hf_export")
os.makedirs(export_dir, exist_ok=True)

# Guardar modelo
model_dir = os.path.join(export_dir, "saved_model")
model.save(model_dir)

# Guardar pesos H5
model.save_weights(os.path.join(export_dir, "model_weights.h5"))

# Copiar tokenizer
shutil.copy(f"{model_prefix}.model", os.path.join(export_dir, "tokenizer.model"))
shutil.copy(f"{model_prefix}.vocab", os.path.join(export_dir, "tokenizer.vocab"))

# Config
model_config = model.get_config()
model_config["model_type"] = "oxide_llm"
with open(os.path.join(export_dir, "config.json"), 'w') as f:
    json.dump(model_config, f, indent=2)

# README
readme = f"""---
license: mit
language:
- es
tags:
- text-generation
- tensorflow
- spanish
- gpt
library_name: tensorflow
---

# OxideLLM

Modelo de lenguaje en español entrenado desde cero.

## Arquitectura
- **Parámetros**: ~{total_params/1e6:.1f}M
- **Vocabulario**: {VOCAB_SIZE} tokens (BPE)
- **Contexto**: {SEQ_LENGTH} tokens
- **Capas**: {config['num_layers']}
- **Dimensión**: {config['d_model']}

## Entrenamiento
Entrenado con literatura clásica española.
"""

with open(os.path.join(export_dir, "README.md"), 'w') as f:
    f.write(readme)

# Subir
REPO_ID = "ULFBERTO/OxideLLM_5M"
try:
    create_repo(REPO_ID, exist_ok=True)
except:
    pass

print(f"📤 Subiendo a {REPO_ID}...")
api.upload_folder(folder_path=export_dir, repo_id=REPO_ID, repo_type="model")
print(f"✅ Subido: https://huggingface.co/{REPO_ID}")


# %% [markdown]
# ## 🔟 Convertir a TFJS y Subir

# %%
import subprocess

# Convertir a TFJS
tfjs_dir = os.path.join(OUTPUT_PATH, "tfjs_model")
os.makedirs(tfjs_dir, exist_ok=True)

print("🔄 Convirtiendo a TensorFlow.js...")

# Intentar conversión
try:
    result = subprocess.run([
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        model_dir,
        tfjs_dir
    ], capture_output=True, text=True, check=True)
    print("✅ Conversión exitosa")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Error: {e.stderr}")
    print("Intentando método alternativo...")
    result = subprocess.run([
        "tensorflowjs_converter",
        "--input_format=keras_saved_model",
        model_dir,
        tfjs_dir
    ], capture_output=True, text=True)

print(f"📁 Archivos TFJS: {os.listdir(tfjs_dir)}")

# %%
# Preparar exportación TFJS
tfjs_export = os.path.join(OUTPUT_PATH, "tfjs_export")
os.makedirs(tfjs_export, exist_ok=True)

# Copiar archivos TFJS
for f in os.listdir(tfjs_dir):
    shutil.copy(os.path.join(tfjs_dir, f), tfjs_export)

# Copiar tokenizer
shutil.copy(f"{model_prefix}.model", os.path.join(tfjs_export, "tokenizer.model"))

# Crear vocab.json para JS
vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
with open(os.path.join(tfjs_export, "vocab.json"), 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

# Config
with open(os.path.join(tfjs_export, "config.json"), 'w') as f:
    json.dump(model_config, f, indent=2)

# README TFJS
tfjs_readme = f"""---
license: mit
language:
- es
tags:
- text-generation
- tfjs
- browser
- spanish
library_name: tensorflowjs
---

# OxideLLM - TensorFlow.js

Versión TFJS para ejecutar en navegador.

## Uso

```javascript
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadGraphModel('model.json');
// Ver documentación completa en el repo principal
```

## Modelo original
[ULFBERTO/OxideLLM_5M](https://huggingface.co/ULFBERTO/OxideLLM_5M)
"""

with open(os.path.join(tfjs_export, "README.md"), 'w') as f:
    f.write(tfjs_readme)

# Subir TFJS
TFJS_REPO = "ULFBERTO/OxideLLM_5M-tfjs"
try:
    create_repo(TFJS_REPO, exist_ok=True)
except:
    pass

print(f"📤 Subiendo TFJS a {TFJS_REPO}...")
api.upload_folder(folder_path=tfjs_export, repo_id=TFJS_REPO, repo_type="model")
print(f"✅ TFJS subido: https://huggingface.co/{TFJS_REPO}")

# %% [markdown]
# ## ✅ ¡Completado!
# 
# Tu modelo está disponible en:
# - **TensorFlow**: https://huggingface.co/ULFBERTO/OxideLLM_5M
# - **TFJS**: https://huggingface.co/ULFBERTO/OxideLLM_5M-tfjs
