"""
Configuración de OxideLearn v2 - Modelo Mejorado
"""

import os

# ============================================================
# RUTAS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
KNOWLEDGE_DB_PATH = os.path.join(DATA_DIR, "knowledge.db")

# Corpus para tokenizer (del proyecto 04)
CORPUS_PATH = os.path.join(BASE_DIR, "..", "Data", "libros_espanol", "corpus_completo.txt")

# ============================================================
# MODELO v2 - ~125M parámetros
# ============================================================
MODEL_CONFIG = {
    "vocab_size": 16000,        # Vocabulario BPE más grande
    "d_model": 768,             # Dimensión del modelo (GPT-2 small)
    "num_heads": 12,            # Cabezas de atención
    "num_layers": 12,           # Capas transformer
    "dff": 3072,                # Dimensión FFN (4x d_model)
    "max_seq_len": 512,         # Longitud máxima
    "dropout": 0.1,
}

# Configuración alternativa más pequeña (~50M) si no tienes suficiente VRAM
MODEL_CONFIG_SMALL = {
    "vocab_size": 16000,
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 8,
    "dff": 2048,
    "max_seq_len": 512,
    "dropout": 0.1,
}

# ============================================================
# MODELO MAESTRO (Llama via Ollama)
# ============================================================
TEACHER_CONFIG = {
    "api_url": "http://localhost:11434/v1",
    "model_name": "llama3.1:8b",
    "temperature": 0.7,
    "max_tokens": 2048,
}

# ============================================================
# ENTRENAMIENTO
# ============================================================
TRAINING_CONFIG = {
    "batch_size": 4,            # Reducido para modelo grande
    "learning_rate": 3e-5,      # Más bajo para modelo grande
    "warmup_steps": 1000,
    "gradient_clip": 1.0,
    "save_every_topics": 50,    # Guardar cada N temas
    "save_every_minutes": 10,   # O cada N minutos
    "max_retries_per_question": 5,  # Intentos máximos por pregunta hasta respuesta correcta
}

# ============================================================
# EJECUCIÓN DE CÓDIGO
# ============================================================
CODE_EXECUTION = {
    "enabled": True,
    "timeout_seconds": 5,
    "allowed_modules": ["math", "random", "datetime", "re", "json"],
}

# ============================================================
# CURRICULUM EXPANDIDO
# ============================================================
CURRICULUM = {
    "matemáticas_básicas": [
        "suma", "resta", "multiplicación", "división",
        "fracciones", "decimales", "porcentajes",
        "potencias", "raíces cuadradas", "orden de operaciones"
    ],
    "matemáticas_avanzadas": [
        "ecuaciones lineales", "ecuaciones cuadráticas",
        "sistemas de ecuaciones", "funciones",
        "trigonometría básica", "logaritmos",
        "probabilidad", "estadística básica"
    ],
    "programación": [
        "variables y tipos de datos", "operadores",
        "condicionales if-else", "bucles for y while",
        "funciones", "listas y arrays",
        "diccionarios", "manejo de strings",
        "recursividad", "algoritmos de ordenamiento"
    ],
    "ciencias": [
        "método científico", "estados de la materia",
        "tabla periódica", "reacciones químicas",
        "leyes de Newton", "energía y trabajo",
        "electricidad básica", "sistema solar",
        "célula y sus partes", "fotosíntesis"
    ],
    "idioma_español": [
        "sustantivos y adjetivos", "verbos regulares",
        "verbos irregulares", "tiempos verbales",
        "pronombres", "preposiciones",
        "acentuación", "signos de puntuación",
        "sinónimos y antónimos", "figuras literarias"
    ],
    "geografía": [
        "continentes y océanos", "capitales del mundo",
        "ríos principales", "montañas y cordilleras",
        "climas del mundo", "países por continente"
    ],
    "historia": [
        "civilizaciones antiguas", "edad media",
        "renacimiento", "revolución industrial",
        "guerras mundiales", "guerra fría"
    ],
    "lógica_razonamiento": [
        "silogismos", "falacias lógicas",
        "operadores lógicos", "tablas de verdad",
        "conjuntos", "diagramas de Venn",
        "razonamiento deductivo", "razonamiento inductivo"
    ]
}

# Crear directorios
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
