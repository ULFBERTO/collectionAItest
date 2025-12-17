import tensorflow as tf
from generate import load_model

def export():
    print("Cargando modelo desde checkpoints...")
    model, char2idx, idx2char = load_model()
    
    if model is None:
        print("No se pudo cargar el modelo.")
        return

    # Guardar en formato SavedModel (carpeta) - Recomendado para TF2
    save_path = './gpt_don_quijote_saved_model'
    print(f"Guardando modelo en formato SavedModel en '{save_path}'...")
    model.save(save_path)
    print("¡Guardado exitoso!")
    
    # Intento de guardar en .keras (nuevo formato de Keras)
    # Nota: Puede requerir versiones recientes de TF/Keras
    try:
        keras_path = 'gpt_don_quijote.keras'
        print(f"Intentando guardar en formato .keras en '{keras_path}'...")
        model.save(keras_path)
        print("¡Guardado exitoso en .keras!")
    except Exception as e:
        print(f"No se pudo guardar en .keras: {e}")

if __name__ == "__main__":
    export()
