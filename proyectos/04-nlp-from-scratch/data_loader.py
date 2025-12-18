import tensorflow as tf
import numpy as np
import os
import requests

# Rutas de datos
CORPUS_PATH = "../Data/libros_espanol/corpus_completo.txt"
DON_QUIJOTE_PATH = "don_quijote.txt"


def download_data(path=None, use_full_corpus=True):
    """
    Carga el texto para entrenamiento.
    
    Args:
        path: Ruta específica al archivo de texto
        use_full_corpus: Si True, usa el corpus completo de libros españoles
    """
    # Determinar qué archivo usar
    if path:
        target_path = path
    elif use_full_corpus and os.path.exists(CORPUS_PATH):
        target_path = CORPUS_PATH
        print(f"📚 Usando corpus completo: {CORPUS_PATH}")
    else:
        target_path = DON_QUIJOTE_PATH
    
    # Si es el corpus completo y existe, cargarlo
    if os.path.exists(target_path):
        print(f"Cargando texto desde {target_path}...")
        with open(target_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"✅ Cargado: {len(text):,} caracteres")
        return text
    
    # Fallback: descargar OxideLLM_5M
    if target_path == DON_QUIJOTE_PATH:
        url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
        print(f"Descargando OxideLLM_5M de {url}...")
        response = requests.get(url)
        text = response.text
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Descarga completada.")
        return text
    
    raise FileNotFoundError(f"No se encontró el archivo: {target_path}")

def preprocess_text(text):
    """Limpia y prepara el texto."""
    # Opcional: Eliminar cabeceras de Gutenberg si se desea una limpieza estricta
    # Por ahora usaremos todo el texto para simplificar
    return text

def create_vocabulary(text):
    """Crea el vocabulario de caracteres únicos."""
    vocab = sorted(set(text))
    print(f'Tamaño del vocabulario: {len(vocab)} caracteres')
    print('Vocabulario:', vocab)
    
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    
    return vocab, char2idx, idx2char

def text_to_int(text, char2idx):
    """Convierte texto a array de enteros."""
    return np.array([char2idx[c] for c in text])

def split_input_target(chunk):
    """Divide una secuencia en entrada y objetivo (desplazado por 1)."""
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def create_dataset(text_as_int, seq_length=100, batch_size=64, buffer_size=10000):
    """Crea un tf.data.Dataset para entrenamiento."""
    # Crear dataset de caracteres
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    
    # Crear secuencias (seq_length + 1 para tener entrada y target)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    
    # Mapear a (entrada, target)
    dataset = sequences.map(split_input_target)
    
    # Shuffle y batch
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # Prueba rápida
    text = download_data()
    vocab, char2idx, idx2char = create_vocabulary(text)
    text_as_int = text_to_int(text, char2idx)
    
    print(f'Texto total: {len(text)} caracteres')
    print(f'Ejemplo codificado: {text[:20]} -> {text_as_int[:20]}')
    
    dataset = create_dataset(text_as_int)
    for input_example, target_example in dataset.take(1):
        print("Input shape:", input_example.shape)
        print("Target shape:", target_example.shape)
        print("Input (chars):", "".join(idx2char[input_example[0].numpy()]))
        print("Target (chars):", "".join(idx2char[target_example[0].numpy()]))
