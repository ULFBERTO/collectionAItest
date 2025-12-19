# OxideLearn 🧠

Sistema de aprendizaje continuo donde un modelo pequeño aprende de un modelo maestro (Llama) bajo demanda.

## Concepto

A diferencia de los LLMs tradicionales que memorizan datos estáticos, OxideLearn:

1. **Detecta** cuando no sabe algo
2. **Pregunta** al modelo maestro (Llama local)
3. **Aprende** el algoritmo/regla/conocimiento
4. **Almacena** en memoria persistente
5. **Aplica** el conocimiento en futuras consultas

## Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│                     OxideLearn                           │
├──────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │   Modelo   │  │  Detector  │  │  Memoria de        │ │
│  │   Base     │  │  de        │  │  Conocimiento      │ │
│  │  (pequeño) │  │  Ignorancia│  │  - Algoritmos      │ │
│  └────────────┘  └────────────┘  │  - Reglas          │ │
│         │              │         │  - Hechos          │ │
│         v              v         └────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐│
│  │              Orquestador de Aprendizaje             ││
│  └─────────────────────────────────────────────────────┘│
│                          │                               │
│                          v                               │
│  ┌─────────────────────────────────────────────────────┐│
│  │         Llama (Maestro) - via LM Studio/Ollama      ││
│  └─────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

## Modos de operación

### 🎓 Modo Aprendizaje
- El modelo puede hacer preguntas a Llama
- Aprende nuevos conceptos y los almacena
- Se ejecuta de forma controlada

### 💬 Modo Chat
- Solo usa conocimiento ya aprendido
- NO aprende de usuarios (seguridad)
- Respuestas rápidas sin consultar maestro

## Requisitos

- Python 3.10+
- LM Studio o Ollama con Llama 3.2 (o similar)
- 8GB+ RAM
- GPU opcional pero recomendada

## Instalación

```bash
cd proyectos/05-oxide-learn
pip install -r requirements.txt
```

## Uso

```bash
# Modo aprendizaje (con Llama como maestro)
python learn.py --topic "matemáticas básicas"
python learn.py --topic "gramática española"
python learn.py --topic "geografía mundial"

# Modo chat (solo inferencia)
python chat.py

# Entrenar modelo base
python train_base.py
```

## Estructura del proyecto

```
05-oxide-learn/
├── config.py           # Configuración global
├── model/
│   ├── base_model.py   # Modelo pequeño (Transformer)
│   ├── memory.py       # Sistema de memoria persistente
│   └── detector.py     # Detector de ignorancia
├── learning/
│   ├── teacher.py      # Conexión con Llama
│   ├── curriculum.py   # Generador de curriculum
│   └── trainer.py      # Entrenamiento continuo
├── inference/
│   └── chat.py         # Modo chat seguro
├── data/
│   ├── knowledge.db    # Base de conocimiento
│   └── checkpoints/    # Checkpoints del modelo
├── learn.py            # Script de aprendizaje
├── chat.py             # Script de chat
└── train_base.py       # Entrenamiento inicial
```

## Tipos de conocimiento

1. **Algoritmos**: Procedimientos paso a paso (matemáticas, lógica)
2. **Reglas**: Patrones gramaticales, sintaxis
3. **Hechos**: Información factual (países, fechas)
4. **Razonamiento**: Cadenas de pensamiento

## Ejemplo de sesión de aprendizaje

```
[OXIDE] ¿Cuánto es 847 × 23?
[OXIDE] No estoy seguro. Consultando al maestro...
[LLAMA] Para multiplicar 847 × 23:
        1. Descompón: 847 × 20 + 847 × 3
        2. 847 × 20 = 16940
        3. 847 × 3 = 2541
        4. 16940 + 2541 = 19481
        El algoritmo es: multiplicación por distribución.
[OXIDE] ✓ Aprendido: algoritmo de multiplicación
[OXIDE] Guardando en memoria...

[OXIDE] ¿Cuánto es 523 × 17?
[OXIDE] Aplicando algoritmo aprendido...
        523 × 10 = 5230
        523 × 7 = 3661
        5230 + 3661 = 8891
[OXIDE] Respuesta: 8891
```

## Licencia

MIT
