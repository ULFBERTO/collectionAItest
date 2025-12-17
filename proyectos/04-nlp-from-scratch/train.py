import tensorflow as tf
import os
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
    
    # 4. Checkpoints
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )
    
    # Intentar cargar checkpoint existente para reanudar
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    initial_epoch = 0
    if latest:
        print(f"Reanudando desde checkpoint: {latest}")
        # Crear variables del modelo con una pasada dummy
        dummy_input = tf.zeros((1, SEQ_LENGTH), dtype=tf.int32)
        model(dummy_input)
        model.load_weights(latest)
        
        # Intentar deducir la época inicial del nombre del archivo (ckpt_X)
        try:
            initial_epoch = int(latest.split('_')[-1])
            print(f"Reanudando en la época {initial_epoch + 1}")
        except:
            print("No se pudo determinar la época exacta, empezando contador en 0 pero con pesos cargados.")

    # 5. Entrenar
    print("Iniciando entrenamiento...")
    history = model.fit(dataset, epochs=EPOCHS, initial_epoch=initial_epoch, callbacks=[checkpoint_callback])
    
    print("Entrenamiento finalizado.")
    return model, char2idx, idx2char

if __name__ == "__main__":
    train()
