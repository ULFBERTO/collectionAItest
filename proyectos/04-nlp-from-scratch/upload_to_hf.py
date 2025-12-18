"""Script simple para subir el modelo guardado a Hugging Face Hub."""

import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "ULFBERTO/gpt-don-quijote"
MODEL_DIR = "./gpt_don_quijote_saved_model"

def upload():
    # Verificar que existe el modelo
    if not os.path.exists(MODEL_DIR):
        print(f"❌ No se encontró el directorio {MODEL_DIR}")
        return
    
    model_file = os.path.join(MODEL_DIR, "model.keras")
    if not os.path.exists(model_file):
        print(f"❌ No se encontró {model_file}")
        return
    
    print(f"📦 Creando repositorio {REPO_ID}...")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Nota: {e}")
    
    print(f"⬆️ Subiendo modelo desde {MODEL_DIR}...")
    api = HfApi()
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model"
    )
    
    print(f"🎉 ¡Modelo subido exitosamente!")
    print(f"🔗 URL: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload()
