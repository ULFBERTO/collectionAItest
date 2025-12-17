import sys
import os
import tensorflow as tf
from data_loader import download_data, create_vocabulary
from model import GPTModel

def resource_path(relative_path):
    """ Obtiene la ruta absoluta al recurso, funciona para dev y PyInstaller """
    try:
        # PyInstaller crea una carpeta temporal en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def generate_text(model, start_string, char2idx, idx2char, num_generate=1000, temperature=1.0, seq_length=100):
    # Convertir string inicial a números
    input_indices = [char2idx.get(s, 0) for s in start_string]
    
    text_generated = list(start_string)
    
    for i in range(num_generate):
        # Usar los últimos seq_length caracteres como contexto
        context = input_indices[-seq_length:]
        
        # Pad al inicio si es necesario
        if len(context) < seq_length:
            context = [0] * (seq_length - len(context)) + context
        
        # Crear tensor de entrada
        input_eval = tf.constant([context], dtype=tf.int32)
        
        predictions = model(input_eval, training=False)
        # Quitar dimensión de batch
        predictions = tf.squeeze(predictions, 0)
        
        # Tomar la predicción del último token
        predictions = predictions[-1] / temperature
        predicted_id = tf.random.categorical(tf.expand_dims(predictions, 0), num_samples=1)[0, 0].numpy()
        
        # Agregar el caracter predicho
        input_indices.append(predicted_id)
        text_generated.append(idx2char[predicted_id])
    
    return ''.join(text_generated)

def load_model():
    """Carga el modelo y los mapeos de caracteres."""
    # Recrear vocabulario
    # Usar resource_path para encontrar el txt en el exe
    txt_path = resource_path('don_quijote.txt')
    text = download_data(txt_path)
    vocab, char2idx, idx2char = create_vocabulary(text)
    
    # 1. Intentar cargar SavedModel (Prioridad para EXE/Distribución)
    saved_model_path = resource_path('gpt_don_quijote_saved_model')
    if os.path.exists(saved_model_path):
        print(f"Cargando SavedModel desde {saved_model_path}...")
        try:
            # Cargar modelo completo
            model = tf.keras.models.load_model(saved_model_path)
            return model, char2idx, idx2char
        except Exception as e:
            print(f"Advertencia: No se pudo cargar SavedModel ({e}). Intentando checkpoints...")

    # 2. Fallback: Cargar desde Checkpoints (Entorno de desarrollo)
    vocab_size = len(vocab)
    
    # Reconstruir modelo
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        dff=512,
        num_layers=4,
        max_len=100,
        dropout=0.1
    )
    
    # Cargar pesos
    checkpoint_dir = './training_checkpoints'
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    
    if latest:
        print(f"Cargando checkpoint: {latest}")
        dummy_input = tf.zeros((1, 100), dtype=tf.int32)
        model(dummy_input)
        model.load_weights(latest).expect_partial()
        return model, char2idx, idx2char
    else:
        print("No se encontró checkpoint.")
        return None, None, None

if __name__ == "__main__":
    model, char2idx, idx2char = load_model()
    if model:
        start_string = "En un lugar de la Mancha"
        print(f"Generando texto con semilla: '{start_string}'...\n")
        generated = generate_text(model, start_string, char2idx, idx2char, num_generate=500)
        print(generated)
