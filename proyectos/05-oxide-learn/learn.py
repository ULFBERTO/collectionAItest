#!/usr/bin/env python3
"""
Script principal de aprendizaje de OxideLearn.
El modelo pequeño aprende del maestro (Llama) sobre temas específicos.

Uso:
    python learn.py --topic "matemáticas básicas"
    python learn.py --topic "capitales de Europa" --type facts
    python learn.py --curriculum matemáticas_básicas
    python learn.py --interactive
"""

import argparse
import os
import sys

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIG, TEACHER_CONFIG, LEARNING_CONFIG,
    KNOWLEDGE_DB_PATH, CHECKPOINTS_DIR, CURRICULUM
)
from model.base_model import create_model, count_parameters
from model.memory import KnowledgeMemory
from model.detector import IgnoranceDetector
from learning.teacher import TeacherLLM, CurriculumGenerator
from learning.trainer import ContinualTrainer


def load_or_create_tokenizer():
    """Carga o crea el tokenizer."""
    try:
        import sentencepiece as spm
        tokenizer_path = os.path.join(CHECKPOINTS_DIR, "tokenizer.model")
        
        if os.path.exists(tokenizer_path):
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            print(f"✅ Tokenizer cargado: {sp.get_piece_size()} tokens")
            return sp
        else:
            print("⚠️ Tokenizer no encontrado. Creando uno...")
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            
            # Buscar corpus existente o crear uno más grande
            corpus_path = os.path.join(CHECKPOINTS_DIR, "temp_train.txt")
            
            # Intentar usar corpus del proyecto 04 si existe
            external_corpus = "../04-nlp-from-scratch/don_quijote.txt"
            alt_corpus = "../../Data/libros_espanol/corpus_completo.txt"
            
            if os.path.exists(external_corpus):
                print(f"   Usando corpus: {external_corpus}")
                import shutil
                shutil.copy(external_corpus, corpus_path)
            elif os.path.exists(alt_corpus):
                print(f"   Usando corpus: {alt_corpus}")
                import shutil
                shutil.copy(alt_corpus, corpus_path)
            else:
                # Crear corpus mínimo pero suficiente
                print("   Creando corpus de entrenamiento...")
                with open(corpus_path, "w", encoding="utf-8") as f:
                    # Texto más extenso para vocabulario
                    f.write(TRAINING_CORPUS)
            
            # Determinar vocab_size basado en tamaño del corpus
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_text = f.read()
            
            # Vocab size máximo = caracteres únicos * 10, mínimo 500
            max_vocab = min(MODEL_CONFIG["vocab_size"], len(set(corpus_text)) * 10)
            max_vocab = max(500, max_vocab)
            
            print(f"   Entrenando tokenizer (vocab_size={max_vocab})...")
            
            model_prefix = os.path.join(CHECKPOINTS_DIR, "tokenizer")
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=max_vocab,
                model_type='bpe',
                character_coverage=0.9995,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3
            )
            
            os.remove(corpus_path)
            
            sp = spm.SentencePieceProcessor()
            sp.load(f"{model_prefix}.model")
            print(f"✅ Tokenizer creado: {sp.get_piece_size()} tokens")
            return sp
            
    except ImportError:
        print("❌ sentencepiece no instalado. Ejecuta: pip install sentencepiece")
        sys.exit(1)


# Corpus de entrenamiento para el tokenizer
TRAINING_CORPUS = """
Hola, ¿cómo estás? Estoy aprendiendo cosas nuevas cada día.
La capital de Francia es París. La capital de España es Madrid.
La capital de Alemania es Berlín. La capital de Italia es Roma.
La capital de Portugal es Lisboa. La capital de Reino Unido es Londres.

Dos más dos es igual a cuatro. Tres por tres es nueve.
Cinco más cinco es diez. Siete por ocho es cincuenta y seis.
El algoritmo de suma consiste en combinar cantidades.
El algoritmo de multiplicación es sumar un número consigo mismo varias veces.
La división es la operación inversa de la multiplicación.
La resta es la operación inversa de la suma.

Los verbos regulares en español terminan en -ar, -er, -ir.
Los verbos irregulares no siguen las reglas de conjugación estándar.
El verbo ser es irregular: soy, eres, es, somos, sois, son.
El verbo estar también es irregular: estoy, estás, está, estamos, estáis, están.
El verbo tener es irregular: tengo, tienes, tiene, tenemos, tenéis, tienen.
El verbo ir es muy irregular: voy, vas, va, vamos, vais, van.

En un lugar de la Mancha, de cuyo nombre no quiero acordarme.
No ha mucho tiempo que vivía un hidalgo de los de lanza en astillero.
Adarga antigua, rocín flaco y galgo corredor.
Una olla de algo más vaca que carnero, salpicón las más noches.
Duelos y quebrantos los sábados, lentejas los viernes.
Algún palomino de añadidura los domingos.

La inteligencia artificial es un campo de la informática.
El aprendizaje automático es una rama de la inteligencia artificial.
Las redes neuronales son modelos inspirados en el cerebro humano.
Los transformers son una arquitectura de red neuronal muy efectiva.
El procesamiento del lenguaje natural permite a las máquinas entender texto.

El sol es una estrella en el centro de nuestro sistema solar.
La Tierra es el tercer planeta del sistema solar.
La Luna es el único satélite natural de la Tierra.
Marte es conocido como el planeta rojo.
Júpiter es el planeta más grande del sistema solar.

El agua está compuesta por hidrógeno y oxígeno.
El aire contiene principalmente nitrógeno y oxígeno.
La fotosíntesis es el proceso por el cual las plantas producen oxígeno.
Los animales respiran oxígeno y exhalan dióxido de carbono.

La historia es el estudio del pasado humano.
La geografía estudia la superficie terrestre y sus habitantes.
La biología es la ciencia que estudia los seres vivos.
La física estudia la materia, la energía y sus interacciones.
La química estudia la composición y propiedades de la materia.
Las matemáticas son el lenguaje universal de la ciencia.

Pregunta: ¿Cuál es la capital de Francia?
Respuesta: La capital de Francia es París.

Pregunta: ¿Cuánto es dos más dos?
Respuesta: Dos más dos es igual a cuatro.

Pregunta: ¿Qué es un verbo irregular?
Respuesta: Un verbo irregular es aquel que no sigue las reglas de conjugación estándar.

Pregunta: ¿Qué es la fotosíntesis?
Respuesta: La fotosíntesis es el proceso por el cual las plantas producen oxígeno usando luz solar.

Usuario: Hola
Asistente: ¡Hola! ¿En qué puedo ayudarte?

Usuario: ¿Qué hora es?
Asistente: No tengo acceso a la hora actual, pero puedo ayudarte con otras preguntas.

Usuario: Gracias
Asistente: ¡De nada! Estoy aquí para ayudar.
"""


def initialize_system():
    """Inicializa todos los componentes del sistema."""
    print("\n" + "=" * 60)
    print("🧠 OxideLearn - Sistema de Aprendizaje Continuo")
    print("=" * 60)
    
    # Tokenizer
    print("\n📝 Cargando tokenizer...")
    tokenizer = load_or_create_tokenizer()
    
    # Modelo
    print("\n🔧 Creando modelo base...")
    model = create_model(MODEL_CONFIG)
    params = count_parameters(model)
    print(f"   Parámetros: {params:,} ({params/1e6:.1f}M)")
    
    # Cargar pesos si existen
    weights_path = os.path.join(CHECKPOINTS_DIR, "model.weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"   ✅ Pesos cargados desde checkpoint")
    
    # Memoria
    print("\n💾 Inicializando memoria...")
    memory = KnowledgeMemory(KNOWLEDGE_DB_PATH)
    stats = memory.get_stats()
    print(f"   Conocimiento almacenado: {stats['total_knowledge']} items")
    
    # Maestro
    print("\n🎓 Conectando con el maestro...")
    teacher = TeacherLLM(**TEACHER_CONFIG)
    
    # Trainer
    trainer = ContinualTrainer(
        model=model,
        tokenizer=tokenizer,
        memory=memory,
        learning_rate=LEARNING_CONFIG["learning_rate"]
    )
    
    # Detector
    detector = IgnoranceDetector(
        confidence_threshold=LEARNING_CONFIG["confidence_threshold"],
        memory=memory
    )
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "memory": memory,
        "teacher": teacher,
        "trainer": trainer,
        "detector": detector
    }


def learn_topic(system: dict, topic: str, knowledge_type: str = "auto"):
    """
    Aprende sobre un tema específico.
    
    Args:
        system: Diccionario con componentes del sistema
        topic: Tema a aprender
        knowledge_type: "algorithm", "facts", "rule", "auto"
    """
    teacher = system["teacher"]
    trainer = system["trainer"]
    memory = system["memory"]
    
    print(f"\n📚 Aprendiendo sobre: {topic}")
    print("-" * 40)
    
    if knowledge_type == "auto":
        # Detectar tipo automáticamente
        if any(word in topic.lower() for word in ["capital", "país", "ciudad", "fecha", "año"]):
            knowledge_type = "facts"
        elif any(word in topic.lower() for word in ["regla", "gramática", "ortografía"]):
            knowledge_type = "rule"
        else:
            knowledge_type = "algorithm"
    
    if knowledge_type == "algorithm":
        print("   Tipo: Algoritmo/Procedimiento")
        algo = teacher.teach_algorithm(topic)
        if algo:
            print(f"   ✅ Algoritmo recibido: {algo.get('name', topic)}")
            loss = trainer.learn_algorithm(algo)
            print(f"   📈 Loss de entrenamiento: {loss:.4f}")
        else:
            print("   ❌ No se pudo obtener el algoritmo")
    
    elif knowledge_type == "facts":
        print("   Tipo: Hechos")
        facts = teacher.teach_facts(topic)
        if facts:
            print(f"   ✅ {len(facts)} hechos recibidos")
            loss = trainer.learn_facts(facts, category=topic)
            print(f"   📈 Loss de entrenamiento: {loss:.4f}")
        else:
            print("   ❌ No se pudieron obtener los hechos")
    
    elif knowledge_type == "rule":
        print("   Tipo: Regla")
        rule = teacher.teach_rule(topic)
        if rule:
            print(f"   ✅ Regla recibida")
            loss = trainer.learn_rule(rule, category=topic)
            print(f"   📈 Loss de entrenamiento: {loss:.4f}")
        else:
            print("   ❌ No se pudo obtener la regla")
    
    # Guardar checkpoint
    trainer.save_checkpoint(CHECKPOINTS_DIR)
    
    # Mostrar estadísticas
    stats = memory.get_stats()
    print(f"\n📊 Memoria actualizada: {stats['total_knowledge']} items")


def learn_curriculum(system: dict, curriculum_name: str):
    """
    Aprende un currículum completo.
    
    Args:
        system: Diccionario con componentes del sistema
        curriculum_name: Nombre del currículum en config.py
    """
    if curriculum_name not in CURRICULUM:
        print(f"❌ Currículum '{curriculum_name}' no encontrado")
        print(f"   Disponibles: {list(CURRICULUM.keys())}")
        return
    
    topics = CURRICULUM[curriculum_name]
    print(f"\n📖 Iniciando currículum: {curriculum_name}")
    print(f"   {len(topics)} temas a aprender")
    print("=" * 60)
    
    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] ", end="")
        learn_topic(system, topic)
    
    print("\n" + "=" * 60)
    print(f"✅ Currículum '{curriculum_name}' completado")
    
    # Estadísticas finales
    stats = system["memory"].get_stats()
    print(f"\n📊 Estadísticas finales:")
    print(f"   Total conocimiento: {stats['total_knowledge']}")
    print(f"   Algoritmos: {stats['algorithms']}")
    print(f"   Reglas: {stats['rules']}")
    print(f"   Hechos: {stats['facts']}")


def interactive_mode(system: dict):
    """
    Modo interactivo de aprendizaje.
    """
    print("\n🎮 Modo Interactivo")
    print("   Comandos:")
    print("   - 'learn <tema>' - Aprender sobre un tema")
    print("   - 'facts <tema>' - Aprender hechos")
    print("   - 'algo <tema>' - Aprender algoritmo")
    print("   - 'rule <tema>' - Aprender regla")
    print("   - 'stats' - Ver estadísticas")
    print("   - 'search <query>' - Buscar en memoria")
    print("   - 'quit' - Salir")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n🧠 > ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if command == "quit" or command == "exit":
                print("👋 ¡Hasta luego!")
                break
            
            elif command == "learn":
                if arg:
                    learn_topic(system, arg)
                else:
                    print("❌ Especifica un tema: learn <tema>")
            
            elif command == "facts":
                if arg:
                    learn_topic(system, arg, "facts")
                else:
                    print("❌ Especifica un tema: facts <tema>")
            
            elif command == "algo":
                if arg:
                    learn_topic(system, arg, "algorithm")
                else:
                    print("❌ Especifica un tema: algo <tema>")
            
            elif command == "rule":
                if arg:
                    learn_topic(system, arg, "rule")
                else:
                    print("❌ Especifica un tema: rule <tema>")
            
            elif command == "stats":
                stats = system["memory"].get_stats()
                print(f"\n📊 Estadísticas:")
                print(f"   Total: {stats['total_knowledge']}")
                print(f"   Por tipo: {stats['by_type']}")
                print(f"   Algoritmos: {stats['algorithms']}")
                print(f"   Reglas: {stats['rules']}")
                print(f"   Hechos: {stats['facts']}")
            
            elif command == "search":
                if arg:
                    results = system["memory"].search_knowledge(arg)
                    if results:
                        print(f"\n🔍 {len(results)} resultados:")
                        for r in results[:5]:
                            print(f"   - [{r['type']}] {r['question'][:50]}...")
                    else:
                        print("   No se encontraron resultados")
                else:
                    print("❌ Especifica búsqueda: search <query>")
            
            else:
                print(f"❌ Comando no reconocido: {command}")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="OxideLearn - Aprendizaje Continuo")
    parser.add_argument("--topic", type=str, help="Tema específico a aprender")
    parser.add_argument("--type", type=str, default="auto",
                       choices=["auto", "algorithm", "facts", "rule"],
                       help="Tipo de conocimiento")
    parser.add_argument("--curriculum", type=str, help="Nombre del currículum")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    parser.add_argument("--list-curricula", action="store_true", help="Listar currículos")
    
    args = parser.parse_args()
    
    if args.list_curricula:
        print("📚 Currículos disponibles:")
        for name, topics in CURRICULUM.items():
            print(f"   - {name}: {len(topics)} temas")
        return
    
    # Inicializar sistema
    system = initialize_system()
    
    if args.interactive:
        interactive_mode(system)
    elif args.curriculum:
        learn_curriculum(system, args.curriculum)
    elif args.topic:
        learn_topic(system, args.topic, args.type)
    else:
        # Sin argumentos, modo interactivo
        interactive_mode(system)


if __name__ == "__main__":
    main()
