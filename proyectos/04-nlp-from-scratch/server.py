"""
FastAPI Server para OxideLLM_5M GPT
Expone el modelo NLP entrenado como API REST con soporte CPU/GPU
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Configurar TensorFlow antes de importarlo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TF

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importar funciones del proyecto existente
from data_loader import download_data, create_vocabulary
from model import GPTModel

# ============== Modelos Pydantic ==============

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Texto inicial para generar")
    num_generate: int = Field(default=300, ge=50, le=1000, description="Cantidad de caracteres a generar")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Creatividad (0.1=conservador, 2.0=creativo)")

class GenerateResponse(BaseModel):
    generated_text: str
    device: str
    prompt: str

class DeviceInfo(BaseModel):
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory: Optional[str] = None

class DeviceRequest(BaseModel):
    device: str = Field(..., pattern="^(cpu|gpu)$", description="'cpu' o 'gpu'")

# ============== Estado Global ==============

class AppState:
    model = None
    char2idx = None
    idx2char = None
    current_device = "cpu"
    gpu_available = False
    gpu_info = None

state = AppState()

# ============== Funciones del Modelo ==============

def detect_gpu():
    """Detecta si hay GPU disponible y obtiene información"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        state.gpu_available = True
        try:
            # Obtener nombre de la GPU
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            state.gpu_info = {
                "name": gpu_details.get('device_name', 'NVIDIA GPU'),
                "memory": "N/A"
            }
        except:
            state.gpu_info = {"name": "NVIDIA GPU", "memory": "N/A"}
        return True
    return False

def set_device(device: str):
    """Configura el dispositivo a usar (cpu o gpu)"""
    if device == "gpu" and not state.gpu_available:
        return False
    
    if device == "cpu":
        # Deshabilitar GPU
        tf.config.set_visible_devices([], 'GPU')
        state.current_device = "cpu"
    else:
        # Habilitar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_visible_devices(gpus, 'GPU')
            state.current_device = "gpu"
    
    return True

def load_model():
    """Carga el modelo y vocabulario"""
    print("📚 Cargando vocabulario...")
    
    # Buscar archivo de texto
    txt_path = Path(__file__).parent / 'don_quijote.txt'
    if not txt_path.exists():
        raise FileNotFoundError(f"No se encontró {txt_path}")
    
    text = download_data(str(txt_path))
    vocab, char2idx, idx2char = create_vocabulary(text)
    
    print("🧠 Cargando modelo...")
    
    # Intentar cargar SavedModel
    saved_model_path = Path(__file__).parent / 'gpt_don_quijote_saved_model'
    if saved_model_path.exists():
        print(f"  → Cargando desde {saved_model_path}")
        try:
            model = tf.keras.models.load_model(str(saved_model_path))
            return model, char2idx, idx2char
        except Exception as e:
            print(f"  ⚠️ Error cargando SavedModel: {e}")
    
    # Fallback: checkpoints
    checkpoint_dir = Path(__file__).parent / 'training_checkpoints'
    latest = tf.train.latest_checkpoint(str(checkpoint_dir))
    
    if latest:
        print(f"  → Cargando desde checkpoint: {latest}")
        vocab_size = len(vocab)
        model = GPTModel(
            vocab_size=vocab_size,
            d_model=256,
            num_heads=4,
            dff=512,
            num_layers=4,
            max_len=100,
            dropout=0.1
        )
        # Build model
        dummy_input = tf.zeros((1, 100), dtype=tf.int32)
        model(dummy_input)
        model.load_weights(latest).expect_partial()
        return model, char2idx, idx2char
    
    raise FileNotFoundError("No se encontró modelo entrenado")

def generate_text(model, start_string: str, char2idx: dict, idx2char: dict, 
                  num_generate: int = 300, temperature: float = 0.8, seq_length: int = 100) -> str:
    """Genera texto usando el modelo"""
    # Convertir string a índices
    input_indices = [char2idx.get(s, 0) for s in start_string]
    text_generated = list(start_string)
    
    for _ in range(num_generate):
        # Contexto de los últimos seq_length caracteres
        context = input_indices[-seq_length:]
        
        # Padding si es necesario
        if len(context) < seq_length:
            context = [0] * (seq_length - len(context)) + context
        
        # Crear tensor
        input_eval = tf.constant([context], dtype=tf.int32)
        
        # Predecir
        predictions = model(input_eval, training=False)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions[-1] / temperature
        
        # Muestrear
        predicted_id = tf.random.categorical(
            tf.expand_dims(predictions, 0), num_samples=1
        )[0, 0].numpy()
        
        # Agregar carácter
        input_indices.append(predicted_id)
        text_generated.append(idx2char[predicted_id])
    
    return ''.join(text_generated)

# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: carga el modelo al iniciar"""
    print("🚀 Iniciando servidor OxideLLM_5M GPT...")
    
    # Detectar GPU
    detect_gpu()
    print(f"  GPU disponible: {state.gpu_available}")
    if state.gpu_available:
        print(f"  GPU: {state.gpu_info['name']}")
        state.current_device = "gpu"
    
    # Cargar modelo
    try:
        state.model, state.char2idx, state.idx2char = load_model()
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise
    
    yield  # App running
    
    print("👋 Cerrando servidor...")

app = FastAPI(
    title="OxideLLM_5M GPT API",
    description="API para generar texto al estilo de El Quijote usando un modelo GPT entrenado desde cero",
    version="1.0.0",
    lifespan=lifespan
)

# CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Endpoints ==============

@app.get("/api/device", response_model=DeviceInfo)
async def get_device():
    """Obtiene información del dispositivo actual"""
    return DeviceInfo(
        device=state.current_device,
        gpu_available=state.gpu_available,
        gpu_name=state.gpu_info["name"] if state.gpu_info else None,
        gpu_memory=state.gpu_info["memory"] if state.gpu_info else None
    )

@app.post("/api/device", response_model=DeviceInfo)
async def change_device(request: DeviceRequest):
    """Cambia el dispositivo de procesamiento"""
    if request.device == "gpu" and not state.gpu_available:
        raise HTTPException(
            status_code=400, 
            detail="GPU no disponible en este sistema"
        )
    
    success = set_device(request.device)
    if not success:
        raise HTTPException(status_code=500, detail="Error cambiando dispositivo")
    
    return await get_device()

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Genera texto a partir de un prompt"""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        generated = generate_text(
            model=state.model,
            start_string=request.prompt,
            char2idx=state.char2idx,
            idx2char=state.idx2char,
            num_generate=request.num_generate,
            temperature=request.temperature
        )
        
        return GenerateResponse(
            generated_text=generated,
            device=state.current_device,
            prompt=request.prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando texto: {str(e)}")

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "device": state.current_device
    }

# ============== Servir Frontend ==============

# Montar archivos estáticos
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """Sirve la página principal"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "OxideLLM_5M GPT API - Visita /docs para la documentación"}

# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
