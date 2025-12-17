import tensorflow as tf
import numpy as np
import os
import requests

def download_data(path='don_quijote.txt'):
    """Descarga el texto de Don Quijote si no existe."""
    if not os.path.exists(path):
        url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
        print(f"Descargando Don Quijote de {url}...")
        response = requests.get(url)
        text = response.text
        
        # Guardar en archivo
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Descarga completada.")
    else:
        print("Archivo ya existe, cargando...")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text

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
