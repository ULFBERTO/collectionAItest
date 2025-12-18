import tensorflow as tf
import os
import json
from data_loader import download_data, create_vocabulary, text_to_int, create_dataset
from model import GPTModel

# Configuración
BATCH_SIZE = 64
BUFFER_SIZE = 10000
SEQ_LENGTH = 100
D_MODEL = 256
NUM_HEADS = 4
DFF = 512
NUM_LAYERS = 4
DROPOUT = 0.1
EPOCHS = 10  # Ajustar según tiempo disponible
CHECKPOINT_DIR = './training_checkpoints'
MODEL_SAVE_DIR = './gpt_don_quijote_saved_model'

# Configuración Hugging Face
HF_REPO_ID = "ULFBERTO/gpt-don-quijote"
UPLOAD_TO_HF = True  # Cambiar a False si no quieres subir automáticamente

def train():
    # 1. Preparar datos
    text = download_data()
    vocab, char2idx, idx2char = create_vocabulary(text)
    vocab_size = len(vocab)
    
    text_as_int = text_to_int(text, char2idx)
    dataset = create_dataset(text_as_int, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
    
    # 2. Crear modelo
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_layers=NUM_LAYERS,
        max_len=SEQ_LENGTH,
        dropout=DROPOUT
    )
    
    # 3. Compilar
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    # 4. Checkpoints (Keras 3 requiere extensión .weights.h5)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch:02d}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )
    
    # Intentar cargar checkpoint existente para reanudar
    initial_epoch = 0
    existing_checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.weights.h5')] if os.path.exists(CHECKPOINT_DIR) else []
    if existing_checkpoints:
        # Ordenar y obtener el más reciente
        existing_checkpoints.sort()
        latest = os.path.join(CHECKPOINT_DIR, existing_checkpoints[-1])
        print(f"Reanudando desde checkpoint: {latest}")
        # Crear variables del modelo con una pasada dummy
        dummy_input = tf.zeros((1, SEQ_LENGTH), dtype=tf.int32)
        model(dummy_input)
        model.load_weights(latest)
        
        # Deducir la época inicial del nombre del archivo (ckpt_XX.weights.h5)
        try:
            initial_epoch = int(existing_checkpoints[-1].split('_')[1].split('.')[0])
            print(f"Reanudando en la época {initial_epoch + 1}")
        except:
            print("No se pudo determinar la época exacta, empezando contador en 0 pero con pesos cargados.")

    # 5. Entrenar
    print("Iniciando entrenamiento...")
    history = model.fit(dataset, epochs=EPOCHS, initial_epoch=initial_epoch, callbacks=[checkpoint_callback])
    
    print("Entrenamiento finalizado.")
    
    # 6. Guardar modelo completo
    print(f"\n💾 Guardando modelo en {MODEL_SAVE_DIR}...")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Guardar modelo (Keras 3 requiere extensión .keras)
    saved_model_path = os.path.join(MODEL_SAVE_DIR, "model.keras")
    model.save(saved_model_path)
    
    # Guardar vocabulario (idx2char es numpy array, convertir a dict)
    vocab_data = {
        "char2idx": char2idx,
        "idx2char": {str(i): char for i, char in enumerate(idx2char)}
    }
    vocab_path = os.path.join(MODEL_SAVE_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    # Guardar configuración
    config = {
        "vocab_size": vocab_size,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "dff": DFF,
        "num_layers": NUM_LAYERS,
        "max_len": SEQ_LENGTH,
        "dropout": DROPOUT
    }
    config_path = os.path.join(MODEL_SAVE_DIR, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Modelo guardado localmente")
    
    return model, char2idx, idx2char, vocab_size


def upload_to_huggingface(repo_id=None, token=None):
    """Sube el modelo entrenado a Hugging Face Hub."""
    try:
        from huggingface_utils import upload_model
        
        repo_id = repo_id or HF_REPO_ID
        
        # Cargar vocab y config
        vocab_path = os.path.join(MODEL_SAVE_DIR, "vocab.json")
        config_path = os.path.join(MODEL_SAVE_DIR, "config.json")
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        upload_model(
            repo_id=repo_id,
            model_dir=MODEL_SAVE_DIR,
            vocab_data=vocab_data,
            config=config,
            token=token
        )
        
    except ImportError:
        print("❌ huggingface_hub no instalado. Ejecuta: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ Error al subir: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Subir a HF después de entrenar")
    parser.add_argument("--repo", type=str, default=None, help="Repo ID de Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Token de Hugging Face")
    args = parser.parse_args()
    
    # Entrenar
    result = train()
    
    # Subir a Hugging Face si se solicita
    if args.upload or UPLOAD_TO_HF:
        print("\n🤗 Subiendo modelo a Hugging Face Hub...")
        upload_to_huggingface(repo_id=args.repo, token=args.token)
