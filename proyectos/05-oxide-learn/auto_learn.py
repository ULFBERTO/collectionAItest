#!/usr/bin/env python3
"""
Aprendizaje autónomo de OxideLearn.
El sistema aprende automáticamente de todos los currículos sin intervención humana.

Uso:
    python auto_learn.py                    # Aprende todo
    python auto_learn.py --curriculum matematicas_basicas  # Solo un currículo
    python auto_learn.py --hours 2          # Aprende por 2 horas
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIG, TEACHER_CONFIG, LEARNING_CONFIG,
    KNOWLEDGE_DB_PATH, CHECKPOINTS_DIR, CURRICULUM
)
from model.base_model import create_model, count_parameters
from model.memory import KnowledgeMemory
from learning.teacher import TeacherLLM, CurriculumGenerator
from learning.trainer import ContinualTrainer


def load_tokenizer():
    """Carga el tokenizer."""
    import sentencepiece as spm
    tokenizer_path = os.path.join(CHECKPOINTS_DIR, "tokenizer.model")
    
    if not os.path.exists(tokenizer_path):
        print("❌ Tokenizer no encontrado. Ejecuta 'python learn.py' primero.")
        sys.exit(1)
    
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp


def initialize_system():
    """Inicializa el sistema."""
    print("\n" + "=" * 60)
    print("🤖 OxideLearn - Aprendizaje Autónomo")
    print("=" * 60)
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tokenizer = load_tokenizer()
    print(f"✅ Tokenizer: {tokenizer.get_piece_size()} tokens")
    
    model = create_model(MODEL_CONFIG)
    weights_path = os.path.join(CHECKPOINTS_DIR, "model.weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("✅ Modelo cargado desde checkpoint")
    else:
        print("✅ Modelo nuevo creado")
    
    params = count_parameters(model)
    print(f"   Parámetros: {params:,} ({params/1e6:.1f}M)")
    
    memory = KnowledgeMemory(KNOWLEDGE_DB_PATH)
    stats = memory.get_stats()
    print(f"✅ Memoria: {stats['total_knowledge']} items previos")
    
    teacher = TeacherLLM(**TEACHER_CONFIG)
    
    trainer = ContinualTrainer(
        model=model,
        tokenizer=tokenizer,
        memory=memory,
        learning_rate=LEARNING_CONFIG["learning_rate"]
    )
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "memory": memory,
        "teacher": teacher,
        "trainer": trainer
    }


def learn_topic_auto(system: dict, topic: str, knowledge_type: str = "auto") -> bool:
    """Aprende un tema automáticamente."""
    teacher = system["teacher"]
    trainer = system["trainer"]
    
    print(f"\n📚 Aprendiendo: {topic}")
    
    # Detectar tipo
    if knowledge_type == "auto":
        if any(w in topic.lower() for w in ["capital", "país", "ciudad", "fecha", "planeta"]):
            knowledge_type = "facts"
        elif any(w in topic.lower() for w in ["regla", "gramática", "ortografía", "conjugación"]):
            knowledge_type = "rule"
        else:
            knowledge_type = "algorithm"
    
    try:
        if knowledge_type == "algorithm":
            print(f"   Tipo: Algoritmo")
            data = teacher.teach_algorithm(topic)
            if data:
                loss = trainer.learn_algorithm(data)
                print(f"   ✅ Aprendido (loss: {loss:.4f})")
                return True
                
        elif knowledge_type == "facts":
            print(f"   Tipo: Hechos")
            data = teacher.teach_facts(topic)
            if data:
                loss = trainer.learn_facts(data, category=topic)
                print(f"   ✅ {len(data)} hechos aprendidos (loss: {loss:.4f})")
                return True
                
        elif knowledge_type == "rule":
            print(f"   Tipo: Regla")
            data = teacher.teach_rule(topic)
            if data:
                loss = trainer.learn_rule(data, category=topic)
                print(f"   ✅ Regla aprendida (loss: {loss:.4f})")
                return True
        
        print(f"   ⚠️ No se pudo obtener información del maestro")
        return False
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def auto_learn_curriculum(system: dict, curriculum_name: str) -> dict:
    """Aprende un currículo completo."""
    if curriculum_name not in CURRICULUM:
        print(f"❌ Currículo '{curriculum_name}' no encontrado")
        return {"success": 0, "failed": 0}
    
    topics = CURRICULUM[curriculum_name]
    print(f"\n{'='*60}")
    print(f"📖 Currículo: {curriculum_name}")
    print(f"   {len(topics)} temas")
    print("="*60)
    
    success = 0
    failed = 0
    
    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}]", end=" ")
        if learn_topic_auto(system, topic):
            success += 1
        else:
            failed += 1
        
        # Pequeña pausa para no sobrecargar
        time.sleep(1)
    
    return {"success": success, "failed": failed}


def auto_learn_all(system: dict, max_hours: float = None):
    """Aprende todos los currículos."""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=max_hours) if max_hours else None
    
    total_success = 0
    total_failed = 0
    curricula_completed = 0
    
    print(f"\n🚀 Iniciando aprendizaje autónomo")
    if end_time:
        print(f"⏱️ Tiempo límite: {max_hours} horas")
    print(f"📚 Currículos a aprender: {len(CURRICULUM)}")
    
    for curriculum_name in CURRICULUM:
        # Verificar tiempo
        if end_time and datetime.now() >= end_time:
            print(f"\n⏰ Tiempo límite alcanzado")
            break
        
        result = auto_learn_curriculum(system, curriculum_name)
        total_success += result["success"]
        total_failed += result["failed"]
        curricula_completed += 1
        
        # Guardar checkpoint después de cada currículo
        system["trainer"].save_checkpoint(CHECKPOINTS_DIR)
        print(f"\n💾 Checkpoint guardado")
    
    # Resumen final
    elapsed = datetime.now() - start_time
    print("\n" + "="*60)
    print("📊 RESUMEN DE APRENDIZAJE")
    print("="*60)
    print(f"⏱️ Tiempo total: {elapsed}")
    print(f"📚 Currículos completados: {curricula_completed}/{len(CURRICULUM)}")
    print(f"✅ Temas aprendidos: {total_success}")
    print(f"❌ Temas fallidos: {total_failed}")
    
    stats = system["memory"].get_stats()
    print(f"\n💾 Estado de la memoria:")
    print(f"   Total conocimiento: {stats['total_knowledge']}")
    print(f"   Algoritmos: {stats['algorithms']}")
    print(f"   Reglas: {stats['rules']}")
    print(f"   Hechos: {stats['facts']}")
    
    return {
        "curricula_completed": curricula_completed,
        "success": total_success,
        "failed": total_failed,
        "elapsed": str(elapsed)
    }


def generate_extra_topics(system: dict, base_topic: str, num_topics: int = 5) -> list:
    """Genera temas adicionales usando el maestro."""
    teacher = system["teacher"]
    
    prompt = f"""Genera {num_topics} subtemas específicos para aprender sobre: {base_topic}

Responde solo con una lista JSON de strings:
["subtema 1", "subtema 2", "subtema 3"]"""
    
    response = teacher.ask(prompt)
    if response:
        import re
        import json
        match = re.search(r'\[[\s\S]*\]', response)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return []


def deep_learn_topic(system: dict, topic: str, depth: int = 2):
    """Aprende un tema en profundidad, generando subtemas."""
    print(f"\n🔬 Aprendizaje profundo: {topic}")
    print(f"   Profundidad: {depth} niveles")
    
    # Aprender tema principal
    learn_topic_auto(system, topic)
    
    if depth > 1:
        # Generar subtemas
        subtopics = generate_extra_topics(system, topic, num_topics=3)
        if subtopics:
            print(f"   📋 Subtemas generados: {len(subtopics)}")
            for subtopic in subtopics:
                deep_learn_topic(system, subtopic, depth - 1)


def main():
    parser = argparse.ArgumentParser(description="OxideLearn - Aprendizaje Autónomo")
    parser.add_argument("--curriculum", type=str, help="Aprender solo este currículo")
    parser.add_argument("--hours", type=float, help="Tiempo máximo en horas")
    parser.add_argument("--topic", type=str, help="Aprender un tema específico en profundidad")
    parser.add_argument("--depth", type=int, default=2, help="Profundidad de aprendizaje")
    parser.add_argument("--list", action="store_true", help="Listar currículos disponibles")
    
    args = parser.parse_args()
    
    if args.list:
        print("📚 Currículos disponibles:")
        for name, topics in CURRICULUM.items():
            print(f"   - {name}: {len(topics)} temas")
        return
    
    # Inicializar
    system = initialize_system()
    
    try:
        if args.topic:
            # Aprendizaje profundo de un tema
            deep_learn_topic(system, args.topic, args.depth)
        elif args.curriculum:
            # Un currículo específico
            auto_learn_curriculum(system, args.curriculum)
        else:
            # Todos los currículos
            auto_learn_all(system, max_hours=args.hours)
        
        # Guardar al final
        system["trainer"].save_checkpoint(CHECKPOINTS_DIR)
        print("\n✅ Aprendizaje completado")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por usuario")
        system["trainer"].save_checkpoint(CHECKPOINTS_DIR)
        print("💾 Progreso guardado")


if __name__ == "__main__":
    main()
