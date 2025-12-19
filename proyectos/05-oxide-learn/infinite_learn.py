#!/usr/bin/env python3
"""
OxideLearn v2 - Entrenamiento Infinito Mejorado
Modelo de ~125M parámetros con ejecución de código.

Uso:
    python infinite_learn.py
    python infinite_learn.py --small  # Usar modelo pequeño (~50M)
"""

import os
import sys
import json
import time
import signal
import random
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIG, MODEL_CONFIG_SMALL, TEACHER_CONFIG, 
    TRAINING_CONFIG, CHECKPOINTS_DIR, DATA_DIR, CORPUS_PATH,
    CODE_EXECUTION, CURRICULUM
)

# Parámetro de reintentos
MAX_RETRIES = TRAINING_CONFIG.get("max_retries_per_question", 100)

# Estado global
RUNNING = True
SAVE_REQUESTED = False


def signal_handler(signum, frame):
    global RUNNING, SAVE_REQUESTED
    print("\n\n⏸️  Pausando... (guardando progreso)")
    RUNNING = False
    SAVE_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)


class OxideTrainerV2:
    """Entrenador mejorado con modelo grande y ejecución de código."""
    
    def __init__(self, use_small_model: bool = False):
        self.use_small_model = use_small_model
        self.model_config = MODEL_CONFIG_SMALL if use_small_model else MODEL_CONFIG
        
        self.state_file = os.path.join(DATA_DIR, "training_state_v2.json")
        self.log_file = os.path.join(DATA_DIR, "training_log_v2.txt")
        
        self.topics_learned = []
        self.total_examples = 0
        self.total_loss_sum = 0
        self.session_start = datetime.now()
        
        self.load_state()
        self._init_components()
    
    def _init_tokenizer(self):
        """Inicializa o crea tokenizer con corpus grande."""
        import sentencepiece as spm
        
        tokenizer_path = os.path.join(CHECKPOINTS_DIR, "tokenizer_v2.model")
        
        if os.path.exists(tokenizer_path):
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            print(f"   ✅ Tokenizer cargado: {sp.get_piece_size()} tokens")
            return sp
        
        print("   🔧 Creando tokenizer con corpus español...")
        
        # Buscar corpus
        if os.path.exists(CORPUS_PATH):
            corpus_file = CORPUS_PATH
            print(f"      Usando: {CORPUS_PATH}")
        else:
            # Crear corpus mínimo
            corpus_file = os.path.join(CHECKPOINTS_DIR, "temp_corpus.txt")
            print(f"      Creando corpus temporal...")
            with open(corpus_file, "w", encoding="utf-8") as f:
                f.write(self._get_training_corpus())
        
        # Entrenar tokenizer
        model_prefix = os.path.join(CHECKPOINTS_DIR, "tokenizer_v2")
        
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=self.model_config["vocab_size"],
            model_type='bpe',
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            num_threads=4,
            max_sentence_length=4096,
        )
        
        if corpus_file.endswith("temp_corpus.txt"):
            os.remove(corpus_file)
        
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")
        print(f"   ✅ Tokenizer creado: {sp.get_piece_size()} tokens")
        
        return sp
    
    def _get_training_corpus(self) -> str:
        """Corpus de entrenamiento para tokenizer."""
        return """
        El conocimiento es poder. La educación es la llave del futuro.
        Las matemáticas son el lenguaje del universo.
        La programación es el arte de resolver problemas.
        
        En un lugar de la Mancha, de cuyo nombre no quiero acordarme,
        no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero,
        adarga antigua, rocín flaco y galgo corredor.
        
        La inteligencia artificial es un campo de la informática que busca
        crear sistemas capaces de realizar tareas que normalmente requieren
        inteligencia humana, como el aprendizaje, el razonamiento y la percepción.
        
        Los transformers son una arquitectura de red neuronal que ha revolucionado
        el procesamiento del lenguaje natural. Utilizan mecanismos de atención
        para procesar secuencias de datos de manera eficiente.
        
        Python es un lenguaje de programación de alto nivel, interpretado y
        de propósito general. Es conocido por su sintaxis clara y legible.
        
        def suma(a, b):
            return a + b
        
        for i in range(10):
            print(i)
        
        if condicion:
            hacer_algo()
        else:
            hacer_otra_cosa()
        
        La capital de España es Madrid. La capital de Francia es París.
        La capital de Alemania es Berlín. La capital de Italia es Roma.
        
        El teorema de Pitágoras establece que en un triángulo rectángulo,
        el cuadrado de la hipotenusa es igual a la suma de los cuadrados
        de los catetos: a² + b² = c²
        
        La fotosíntesis es el proceso por el cual las plantas convierten
        la luz solar, el agua y el dióxido de carbono en glucosa y oxígeno.
        
        Pregunta: ¿Cuánto es 2 + 2?
        Respuesta: 2 + 2 = 4
        
        Pregunta: ¿Cuál es la capital de Francia?
        Respuesta: La capital de Francia es París.
        
        Usuario: Hola, ¿cómo estás?
        Asistente: ¡Hola! Estoy bien, gracias por preguntar. ¿En qué puedo ayudarte?
        
        Usuario: Explícame qué es una función en programación.
        Asistente: Una función es un bloque de código reutilizable que realiza una tarea específica.
        Se define una vez y se puede llamar múltiples veces desde diferentes partes del programa.
        """ * 10  # Repetir para tener suficiente texto
    
    def _init_components(self):
        """Inicializa todos los componentes."""
        import tensorflow as tf
        
        print("\n" + "=" * 60)
        print("🔧 Inicializando OxideLearn v2")
        print("=" * 60)
        
        model_type = "pequeño (~50M)" if self.use_small_model else "grande (~125M)"
        print(f"   Modelo: {model_type}")
        
        # Tokenizer
        print("\n📝 Tokenizer:")
        self.tokenizer = self._init_tokenizer()
        
        # Actualizar vocab_size real
        actual_vocab = self.tokenizer.get_piece_size()
        self.model_config = self.model_config.copy()
        self.model_config["vocab_size"] = actual_vocab
        
        # Modelo
        print("\n🧠 Modelo:")
        from model.base_model import create_model, count_parameters
        
        self.model = create_model(self.model_config)
        params = count_parameters(self.model)
        print(f"   ✅ Parámetros: {params:,} ({params/1e6:.1f}M)")
        print(f"   d_model: {self.model_config['d_model']}")
        print(f"   Capas: {self.model_config['num_layers']}")
        print(f"   Cabezas: {self.model_config['num_heads']}")
        
        # Cargar pesos
        weights_path = os.path.join(CHECKPOINTS_DIR, "model_v2.weights.h5")
        if os.path.exists(weights_path):
            try:
                self.model.load_weights(weights_path)
                print(f"   ✅ Pesos cargados desde checkpoint")
            except Exception as e:
                print(f"   ⚠️ No se pudieron cargar pesos: {e}")
        
        # Optimizer con warmup
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=TRAINING_CONFIG["learning_rate"],
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Maestro
        print("\n🎓 Maestro:")
        from learning.teacher import TeacherLLM
        self.teacher = TeacherLLM(**TEACHER_CONFIG)
        
        # Ejecutor de código
        print("\n💻 Ejecutor de código:")
        if CODE_EXECUTION["enabled"]:
            from tools.code_executor import SafeCodeExecutor
            self.code_executor = SafeCodeExecutor(timeout=CODE_EXECUTION["timeout_seconds"])
            print("   ✅ Habilitado")
        else:
            self.code_executor = None
            print("   ❌ Deshabilitado")
        
        # Memoria
        print("\n💾 Memoria:")
        from model.memory import KnowledgeMemory
        from config import KNOWLEDGE_DB_PATH
        self.memory = KnowledgeMemory(KNOWLEDGE_DB_PATH)
        stats = self.memory.get_stats()
        print(f"   ✅ {stats['total_knowledge']} items en memoria")
    
    def load_state(self):
        """Carga estado previo."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.topics_learned = state.get("topics_learned", [])
                self.total_examples = state.get("total_examples", 0)
                self.total_loss_sum = state.get("total_loss_sum", 0)
                print(f"\n📂 Estado anterior:")
                print(f"   Temas: {len(self.topics_learned)}")
                print(f"   Ejemplos: {self.total_examples}")
            except:
                pass
    
    def save_state(self):
        """Guarda estado."""
        state = {
            "topics_learned": self.topics_learned[-2000:],
            "total_examples": self.total_examples,
            "total_loss_sum": self.total_loss_sum,
            "last_save": datetime.now().isoformat(),
            "model_config": self.model_config
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def save_checkpoint(self):
        """Guarda modelo y estado."""
        weights_path = os.path.join(CHECKPOINTS_DIR, "model_v2.weights.h5")
        self.model.save_weights(weights_path)
        self.save_state()
        print(f"💾 Checkpoint guardado")
    
    def generate_topic(self) -> str:
        """Genera tema para aprender."""
        # 70% del curriculum, 30% generado por Llama
        if random.random() < 0.7:
            # Del curriculum
            category = random.choice(list(CURRICULUM.keys()))
            topic = random.choice(CURRICULUM[category])
            return f"{topic}"
        else:
            # Generado por Llama
            prompt = """Genera UN tema específico para aprender. Puede ser sobre:
matemáticas, ciencia, programación, idiomas, historia, geografía, lógica.

Responde SOLO con el tema, sin explicación. Ejemplo: "Derivadas parciales"

Tema:"""
            response = self.teacher.ask(prompt)
            if response:
                topic = response.strip().split("\n")[0].strip()
                if 5 < len(topic) < 100:
                    return topic
        
        # Fallback
        return random.choice([
            "ecuaciones de segundo grado",
            "algoritmo de ordenamiento",
            "conjugación verbal",
            "leyes de Newton"
        ])
    
    def generate_response(self, prompt: str, max_tokens: int = 100) -> str:
        """Genera respuesta con el modelo Oxide."""
        import tensorflow as tf
        
        tokens = self.tokenizer.encode(prompt)
        
        for _ in range(max_tokens):
            input_ids = tf.constant([tokens[-self.model_config["max_seq_len"]:]], dtype=tf.int32)
            logits = self.model(input_ids, training=False)
            logits = logits[0, -1, :] / 0.8  # temperature
            
            # Top-k sampling
            top_k = 40
            top_logits, top_indices = tf.math.top_k(logits, k=top_k)
            probs = tf.nn.softmax(top_logits)
            next_idx = tf.random.categorical(tf.expand_dims(tf.math.log(probs + 1e-10), 0), 1)
            next_token = int(top_indices[next_idx[0, 0]].numpy())
            
            tokens.append(next_token)
            
            # Stop en EOS o newline
            if next_token == 3:
                break
            
            decoded = self.tokenizer.decode([next_token])
            if '\n' in decoded and len(tokens) > len(prompt) + 10:
                break
        
        return self.tokenizer.decode(tokens)
    
    def learn_topic(self, topic: str, max_retries: int = MAX_RETRIES) -> bool:
        """Aprende un tema con ciclo de evaluación y re-entrenamiento hasta respuesta correcta."""
        import tensorflow as tf
        import re
        
        print(f"\n{'='*60}")
        print(f"📚 TEMA: {topic}")
        print(f"{'='*60}")
        
        # FASE 1: Obtener ejemplos del maestro
        print(f"\n🎓 [FASE 1] Maestro genera ejemplos...")
        
        prompt = f"""Enseña sobre: {topic}

Genera 5 pares de pregunta-respuesta para entrenar un modelo.
Las respuestas deben ser claras, completas y educativas.

Formato JSON:
[
    {{"pregunta": "...", "respuesta": "..."}},
]

Solo JSON, sin texto adicional."""
        
        response = self.teacher.ask(prompt)
        
        if not response:
            print("   ⚠️ Sin respuesta")
            return False
        
        # Parsear
        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                examples = json.loads(json_match.group())
            else:
                print("   ⚠️ No se encontró JSON")
                return False
        except Exception as e:
            print(f"   ⚠️ Error JSON: {e}")
            return False
        
        if len(examples) < 2:
            print("   ⚠️ Pocos ejemplos")
            return False
        
        print(f"   ✅ {len(examples)} ejemplos recibidos")
        
        # FASE 2: Entrenamiento inicial
        print(f"\n🧠 [FASE 2] Entrenamiento inicial...")
        
        total_loss = 0
        num_trained = 0
        
        for ex in examples:
            loss = self._train_on_example(ex, tf)
            if loss is not None:
                total_loss += loss
                num_trained += 1
        
        if num_trained == 0:
            return False
        
        avg_loss = total_loss / num_trained
        print(f"   ✅ Loss inicial: {avg_loss:.4f}")
        
        # FASE 3 y 4: Evaluación con ciclo de re-entrenamiento
        print(f"\n📝 [FASE 3-4] Evaluación con ciclo de corrección (max {max_retries} intentos)...")
        
        total_corrections = 0
        total_correct = 0
        
        for i, ex in enumerate(examples[:3]):  # Evaluar 3 ejemplos
            pregunta = ex.get("pregunta", "")
            respuesta_correcta = ex.get("respuesta", "")
            
            if not pregunta:
                continue
            
            print(f"\n   ┌─ [{i+1}] {pregunta[:60]}...")
            
            # Ciclo de re-entrenamiento hasta respuesta correcta
            is_correct = False
            retry = 0
            
            while not is_correct and retry < max_retries and RUNNING:
                retry += 1
                
                # Oxide intenta responder
                prompt_oxide = f"Pregunta: {pregunta}\nRespuesta:"
                respuesta_oxide = self.generate_response(prompt_oxide, max_tokens=80)
                
                # Extraer solo la respuesta
                if "Respuesta:" in respuesta_oxide:
                    respuesta_oxide = respuesta_oxide.split("Respuesta:")[-1].strip()
                respuesta_oxide = respuesta_oxide[:200]
                
                print(f"   │  Intento {retry}: {respuesta_oxide[:500]}...")
                
                # Maestro evalúa
                eval_prompt = f"""Evalúa esta respuesta de un estudiante:

Pregunta: {pregunta}
Respuesta del estudiante: {respuesta_oxide}
Respuesta correcta: {respuesta_correcta}

Responde en JSON:
{{"correcto": true/false, "feedback": "explicación breve", "correccion": "respuesta mejorada si es incorrecta"}}

Solo JSON:"""
                
                eval_response = self.teacher.ask(eval_prompt)
                
                if eval_response:
                    try:
                        json_match = re.search(r'\{[\s\S]*\}', eval_response)
                        if json_match:
                            evaluation = json.loads(json_match.group())
                            is_correct = evaluation.get("correcto", False)
                            feedback = evaluation.get("feedback", "")
                            correccion = evaluation.get("correccion", respuesta_correcta)
                            
                            if is_correct:
                                print(f"   │  ✅ ¡Correcto en intento {retry}!")
                                total_correct += 1
                            else:
                                print(f"   │  ❌ {feedback[:40]}...")
                                
                                # Entrenar con la corrección
                                correction_ex = {
                                    "pregunta": pregunta,
                                    "respuesta": correccion
                                }
                                
                                # Entrenar múltiples veces para reforzar
                                reinforcement_rounds = 5 if retry > 2 else 3
                                for _ in range(reinforcement_rounds):
                                    self._train_on_example(correction_ex, tf)
                                
                                total_corrections += 1
                                print(f"   │  🔧 Entrenado ({reinforcement_rounds}x refuerzo)")
                    except Exception as e:
                        print(f"   │  ⚠️ Error evaluación: {e}")
                        break
                else:
                    print(f"   │  ⚠️ Sin respuesta del maestro")
                    break
            
            if not is_correct and retry >= max_retries:
                print(f"   │  ⚠️ Max intentos alcanzado, continuando...")
                # Entrenar una última vez con la respuesta correcta
                final_ex = {"pregunta": pregunta, "respuesta": respuesta_correcta}
                for _ in range(5):
                    self._train_on_example(final_ex, tf)
            
            print(f"   └─ Completado")
        
        # Actualizar estadísticas
        self.total_examples += num_trained
        self.total_loss_sum += avg_loss
        self.topics_learned.append(topic)
        
        print(f"\n✅ TEMA COMPLETADO")
        print(f"   Loss inicial: {avg_loss:.4f}")
        print(f"   Ejemplos: {num_trained}")
        print(f"   Correctos: {total_correct}/3")
        print(f"   Correcciones aplicadas: {total_corrections}")
        
        # Log
        self._log(topic, examples, avg_loss)
        
        return True
    
    def _train_on_example(self, ex: dict, tf) -> float:
        """Entrena el modelo con un ejemplo. Retorna loss o None si falla."""
        pregunta = ex.get("pregunta", "")
        respuesta = ex.get("respuesta", "")
        
        if not pregunta or not respuesta:
            return None
        
        full_text = f"Pregunta: {pregunta}\nRespuesta: {respuesta}"
        tokens = self.tokenizer.encode(full_text)
        
        if len(tokens) < 10:
            return None
        
        max_len = self.model_config["max_seq_len"]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        input_ids = tf.constant([tokens[:-1]], dtype=tf.int32)
        target_ids = tf.constant([tokens[1:]], dtype=tf.int32)
        
        with tf.GradientTape() as tape:
            logits = self.model(input_ids, training=True)
            min_len = min(logits.shape[1], target_ids.shape[1])
            loss = self.loss_fn(target_ids[:, :min_len], logits[:, :min_len, :])
        
        gradients = tape.gradient(loss, self.model.trainable_weights)
        if gradients[0] is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, TRAINING_CONFIG["gradient_clip"])
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        return float(loss)
    


    
    def _log(self, topic: str, examples: list, loss: float):
        """Guarda log."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.now().isoformat()}] {topic}\n")
            f.write(f"Loss: {loss:.4f}\n")
            for ex in examples[:3]:
                f.write(f"  P: {ex.get('pregunta', '')[:100]}\n")
                f.write(f"  R: {ex.get('respuesta', '')[:100]}\n")
    
    def run(self):
        """Ejecuta entrenamiento infinito."""
        global RUNNING, SAVE_REQUESTED
        
        print("\n" + "=" * 60)
        print("🔄 ENTRENAMIENTO INFINITO v2")
        print("=" * 60)
        print("   Ctrl+C para pausar y guardar")
        print(f"   Auto-guardado cada {TRAINING_CONFIG['save_every_topics']} temas")
        print("-" * 60)
        
        topics_this_session = 0
        last_save = time.time()
        
        while RUNNING:
            topic = self.generate_topic()
            
            # Evitar repetir
            if topic in self.topics_learned[-100:]:
                continue
            
            success = self.learn_topic(topic)
            
            if success:
                topics_this_session += 1
            
            # Auto-guardar
            should_save = (
                topics_this_session % TRAINING_CONFIG["save_every_topics"] == 0 or
                (time.time() - last_save) > TRAINING_CONFIG["save_every_minutes"] * 60
            )
            
            if should_save and topics_this_session > 0:
                self.save_checkpoint()
                last_save = time.time()
                
                elapsed = datetime.now() - self.session_start
                avg_loss = self.total_loss_sum / max(len(self.topics_learned), 1)
                
                print(f"\n📊 PROGRESO")
                print(f"   Sesión: {topics_this_session} temas | {elapsed}")
                print(f"   Total histórico: {len(self.topics_learned)} temas")
                print(f"   Loss promedio: {avg_loss:.4f}")
            
            time.sleep(0.5)
        
        # Guardar al salir
        if SAVE_REQUESTED:
            self.save_checkpoint()
            
            elapsed = datetime.now() - self.session_start
            print("\n" + "=" * 60)
            print("⏸️  PAUSADO")
            print("=" * 60)
            print(f"   Duración: {elapsed}")
            print(f"   Temas sesión: {topics_this_session}")
            print(f"   Total: {len(self.topics_learned)}")
            print(f"\n   Ejecuta 'python infinite_learn.py' para continuar")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Usar modelo pequeño (~50M)")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🧠 OxideLearn v2 - Entrenamiento Infinito")
    print("=" * 60)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    trainer = OxideTrainerV2(use_small_model=args.small)
    trainer.run()


if __name__ == "__main__":
    main()
