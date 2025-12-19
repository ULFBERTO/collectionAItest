# ============================================================
# SECCIÓN 1: SETUP - Ejecutar primero en Google Colab
# ============================================================
# ⚠️ IMPORTANTE: Antes de ejecutar, activa GPU en Colab:
#    Runtime > Change runtime type > GPU (T4)
# ============================================================

# Verificar GPU
!nvidia-smi

# Si dice "command not found", ve a Runtime > Change runtime type > GPU

# Instalar dependencias (usar versión de TF que ya tiene Colab)
!pip install -q sentencepiece huggingface_hub tensorflowjs --upgrade

# Verificar TensorFlow
import tensorflow as tf
print(f"✅ TensorFlow: {tf.__version__}")
print(f"✅ GPU disponible: {tf.config.list_physical_devices('GPU')}")

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Configuración de rutas
import os

# Cambiar esta ruta según tu estructura en Drive
DATA_PATH = "/content/drive/MyDrive/OxideLLM/libros_espanol"
OUTPUT_PATH = "/content/drive/MyDrive/OxideLLM/checkpoints"
TOKENIZER_PATH = "/content/drive/MyDrive/OxideLLM/tokenizer"

# Crear directorios
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(TOKENIZER_PATH, exist_ok=True)

print("✅ Setup completado")
print(f"📁 Datos: {DATA_PATH}")
print(f"💾 Checkpoints: {OUTPUT_PATH}")
