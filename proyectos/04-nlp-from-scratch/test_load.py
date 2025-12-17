import tensorflow as tf
from model import GPTModel, PositionalEmbedding, CausalSelfAttention, FeedForward, TransformerBlock
from data_loader import download_data, create_vocabulary, text_to_int

def test_load():
    save_path = './gpt_don_quijote_saved_model'
    print(f"Cargando modelo desde '{save_path}'...")
    
    try:
        # Cargar modelo
        # Es posible que necesitemos pasar custom_objects si no se registraron automáticamente
        model = tf.keras.models.load_model(save_path, custom_objects={
            'GPTModel': GPTModel,
            'PositionalEmbedding': PositionalEmbedding,
            'CausalSelfAttention': CausalSelfAttention,
            'FeedForward': FeedForward,
            'TransformerBlock': TransformerBlock
        })
        print("¡Modelo cargado exitosamente!")
        
        # Probar inferencia
        text = download_data()
        vocab, char2idx, idx2char = create_vocabulary(text)
        
        start_string = "Sancho"
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        
        print("Generando prueba...")
        prediction = model(input_eval)
        print("Output shape:", prediction.shape)
        print("¡Inferencia exitosa!")
        
    except Exception as e:
        print(f"Error al cargar: {e}")

if __name__ == "__main__":
    test_load()
