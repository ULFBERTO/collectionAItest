"""
Entrenador continuo para OxideLearn.
Permite que el modelo aprenda de forma incremental sin olvidar.
"""

import tensorflow as tf
import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class ContinualTrainer:
    """
    Entrenador que permite aprendizaje continuo.
    
    Técnicas anti-olvido:
    1. Elastic Weight Consolidation (EWC)
    2. Replay de ejemplos anteriores
    3. Regularización de conocimiento previo
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        memory,
        learning_rate: float = 1e-4,
        ewc_lambda: float = 0.4,
        replay_ratio: float = 0.3
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda
        self.replay_ratio = replay_ratio
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Para EWC: importancia de cada peso
        self.fisher_information = None
        self.optimal_weights = None
        
        # Buffer de replay
        self.replay_buffer = []
        self.max_replay_size = 1000
        
        # Historial
        self.training_history = []
    
    def _tokenize(self, text: str, max_len: int = 512) -> tf.Tensor:
        """Tokeniza texto."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))
        return tf.constant([tokens], dtype=tf.int32)
    
    def _create_training_pair(self, input_text: str, output_text: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """Crea par de entrenamiento (input, target)."""
        full_text = f"{input_text} {output_text}"
        tokens = self.tokenizer.encode(full_text)
        
        max_len = self.model.max_seq_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        input_ids = tf.constant([tokens[:-1]], dtype=tf.int32)
        target_ids = tf.constant([tokens[1:]], dtype=tf.int32)
        
        return input_ids, target_ids
    
    @tf.function
    def _train_step(self, input_ids: tf.Tensor, target_ids: tf.Tensor) -> tf.Tensor:
        """Paso de entrenamiento."""
        with tf.GradientTape() as tape:
            logits = self.model(input_ids, training=True)
            
            # Loss principal
            loss = self.loss_fn(target_ids, logits)
            
            # EWC regularization
            if self.fisher_information is not None and self.optimal_weights is not None:
                ewc_loss = 0.0
                for i, weight in enumerate(self.model.trainable_weights):
                    ewc_loss += tf.reduce_sum(
                        self.fisher_information[i] * 
                        tf.square(weight - self.optimal_weights[i])
                    )
                loss += self.ewc_lambda * ewc_loss
        
        gradients = tape.gradient(loss, self.model.trainable_weights)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        return loss
    
    def learn_from_example(self, input_text: str, output_text: str) -> float:
        """
        Aprende de un único ejemplo.
        
        Args:
            input_text: Pregunta o entrada
            output_text: Respuesta o salida esperada
        
        Returns:
            Loss del entrenamiento
        """
        input_ids, target_ids = self._create_training_pair(input_text, output_text)
        loss = self._train_step(input_ids, target_ids)
        
        # Agregar al buffer de replay
        self._add_to_replay(input_text, output_text)
        
        return float(loss)
    
    def learn_batch(self, examples: List[Dict], epochs: int = 1) -> List[float]:
        """
        Aprende de un batch de ejemplos.
        
        Args:
            examples: Lista de {"input": str, "output": str}
            epochs: Número de épocas
        
        Returns:
            Lista de losses por época
        """
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Mezclar ejemplos nuevos con replay
            training_examples = examples.copy()
            if self.replay_buffer and self.replay_ratio > 0:
                num_replay = int(len(examples) * self.replay_ratio)
                replay_samples = np.random.choice(
                    len(self.replay_buffer),
                    min(num_replay, len(self.replay_buffer)),
                    replace=False
                )
                for idx in replay_samples:
                    training_examples.append(self.replay_buffer[idx])
            
            np.random.shuffle(training_examples)
            
            for example in training_examples:
                loss = self.learn_from_example(example["input"], example["output"])
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            print(f"  Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Actualizar Fisher Information después de aprender
        self._update_fisher_information(examples)
        
        return losses
    
    def _add_to_replay(self, input_text: str, output_text: str):
        """Agrega ejemplo al buffer de replay."""
        self.replay_buffer.append({"input": input_text, "output": output_text})
        
        # Mantener tamaño máximo
        if len(self.replay_buffer) > self.max_replay_size:
            # Eliminar ejemplos antiguos aleatorios
            remove_idx = np.random.randint(0, len(self.replay_buffer) - 100)
            self.replay_buffer.pop(remove_idx)
    
    def _update_fisher_information(self, examples: List[Dict]):
        """
        Actualiza la Fisher Information Matrix para EWC.
        Esto ayuda a preservar conocimiento importante.
        """
        if len(examples) < 5:
            return
        
        # Calcular gradientes cuadrados promedio
        fisher = [tf.zeros_like(w) for w in self.model.trainable_weights]
        
        for example in examples[:50]:  # Limitar para eficiencia
            input_ids, target_ids = self._create_training_pair(
                example["input"], example["output"]
            )
            
            with tf.GradientTape() as tape:
                logits = self.model(input_ids, training=False)
                loss = self.loss_fn(target_ids, logits)
            
            gradients = tape.gradient(loss, self.model.trainable_weights)
            
            for i, grad in enumerate(gradients):
                if grad is not None:
                    fisher[i] += tf.square(grad)
        
        # Normalizar
        num_examples = min(len(examples), 50)
        fisher = [f / num_examples for f in fisher]
        
        # Combinar con Fisher anterior si existe
        if self.fisher_information is not None:
            fisher = [
                0.5 * old + 0.5 * new 
                for old, new in zip(self.fisher_information, fisher)
            ]
        
        self.fisher_information = fisher
        self.optimal_weights = [tf.Variable(w) for w in self.model.trainable_weights]
    
    def learn_algorithm(self, algorithm: Dict) -> float:
        """
        Aprende un algoritmo del maestro.
        
        Args:
            algorithm: Dict con name, description, steps, examples
        """
        examples = []
        
        # Crear ejemplos de entrenamiento del algoritmo
        name = algorithm.get("name", "algoritmo")
        description = algorithm.get("description", "")
        steps = algorithm.get("steps", [])
        algo_examples = algorithm.get("examples", [])
        
        # Ejemplo: explicar el algoritmo
        input_text = f"¿Cómo funciona {name}?"
        output_text = f"{description}. Pasos: " + " ".join(
            f"{i+1}. {step}" for i, step in enumerate(steps)
        )
        examples.append({"input": input_text, "output": output_text})
        
        # Ejemplos prácticos
        for ex in algo_examples:
            input_text = f"Aplica {name}: {ex.get('input', '')}"
            output_text = ex.get("output", "") 
            if ex.get("explanation"):
                output_text += f" ({ex['explanation']})"
            examples.append({"input": input_text, "output": output_text})
        
        # Entrenar
        losses = self.learn_batch(examples, epochs=3)
        
        # Guardar en memoria
        self.memory.store_algorithm(
            name=name,
            description=description,
            steps=steps,
            examples=algo_examples
        )
        
        return np.mean(losses)
    
    def learn_facts(self, facts: List[Dict], category: str = "general") -> float:
        """
        Aprende hechos del maestro.
        
        Args:
            facts: Lista de {"subject", "predicate", "object"}
            category: Categoría de los hechos
        """
        examples = []
        
        for fact in facts:
            subject = fact.get("subject", "")
            predicate = fact.get("predicate", "")
            obj = fact.get("object", "")
            
            # Crear variaciones de preguntas
            if "capital" in predicate.lower():
                questions = [
                    f"¿Cuál es la capital de {subject}?",
                    f"Capital de {subject}",
                    f"{subject} capital"
                ]
                answer = obj
            elif "es" in predicate.lower() or "tiene" in predicate.lower():
                questions = [
                    f"¿Qué {predicate} {subject}?",
                    f"{subject} {predicate}"
                ]
                answer = obj
            else:
                questions = [f"¿{predicate} de {subject}?"]
                answer = obj
            
            for q in questions:
                examples.append({"input": q, "output": answer})
            
            # Guardar en memoria
            self.memory.store_fact(category, subject, predicate, obj)
        
        # Entrenar
        losses = self.learn_batch(examples, epochs=2)
        return np.mean(losses)
    
    def learn_rule(self, rule: Dict, category: str = "general") -> float:
        """
        Aprende una regla del maestro.
        
        Args:
            rule: Dict con rule_text, exceptions, examples
            category: Categoría de la regla
        """
        examples = []
        
        rule_text = rule.get("rule_text", "")
        exceptions = rule.get("exceptions", [])
        rule_examples = rule.get("examples", [])
        
        # Pregunta sobre la regla
        examples.append({
            "input": f"¿Cuál es la regla de {category}?",
            "output": rule_text
        })
        
        # Excepciones
        if exceptions:
            examples.append({
                "input": f"¿Hay excepciones a la regla de {category}?",
                "output": "Sí, las excepciones son: " + ", ".join(exceptions)
            })
        
        # Ejemplos
        for ex in rule_examples:
            examples.append({
                "input": f"Dame un ejemplo de {category}",
                "output": ex
            })
        
        # Entrenar
        losses = self.learn_batch(examples, epochs=2)
        
        # Guardar en memoria
        self.memory.store_rule(category, rule_text, exceptions, rule_examples)
        
        return np.mean(losses)
    
    def save_checkpoint(self, path: str):
        """Guarda checkpoint del entrenamiento."""
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelo
        self.model.save_weights(os.path.join(path, "model.weights.h5"))
        
        # Guardar estado del trainer
        state = {
            "replay_buffer": self.replay_buffer[-500:],  # Últimos 500
            "training_history": self.training_history,
            "timestamp": datetime.now().isoformat()
        }
        with open(os.path.join(path, "trainer_state.json"), "w") as f:
            json.dump(state, f)
        
        print(f"✅ Checkpoint guardado en {path}")
    
    def load_checkpoint(self, path: str):
        """Carga checkpoint."""
        weights_path = os.path.join(path, "model.weights.h5")
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"✅ Pesos cargados desde {weights_path}")
        
        state_path = os.path.join(path, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self.replay_buffer = state.get("replay_buffer", [])
            self.training_history = state.get("training_history", [])
            print(f"✅ Estado del trainer cargado")


if __name__ == "__main__":
    print("Trainer module loaded. Use with OxideLearn system.")
