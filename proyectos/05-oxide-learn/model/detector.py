"""
Detector de ignorancia para OxideLearn.
Determina cuándo el modelo no sabe algo y necesita aprender.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Optional
import re


class IgnoranceDetector:
    """
    Detecta cuándo el modelo no sabe responder correctamente.
    
    Métodos de detección:
    1. Confianza del modelo (capa de confianza)
    2. Entropía de la distribución de salida
    3. Patrones de respuesta evasiva
    4. Verificación con conocimiento almacenado
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        entropy_threshold: float = 3.0,
        memory=None
    ):
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.memory = memory
        
        # Patrones que indican que el modelo no sabe
        self.uncertainty_patterns = [
            r"no\s+(sé|estoy\s+seguro)",
            r"no\s+tengo\s+(información|datos)",
            r"desconozco",
            r"no\s+puedo\s+(responder|decir)",
            r"necesito\s+más\s+información",
            r"no\s+entiendo",
            r"¿qué\s+es\s+eso\?",
            r"creo\s+que",  # Indica incertidumbre
            r"tal\s+vez",
            r"quizás?",
            r"probablemente",
        ]
        self.uncertainty_regex = re.compile(
            "|".join(self.uncertainty_patterns), 
            re.IGNORECASE
        )
    
    def calculate_entropy(self, logits: tf.Tensor) -> float:
        """Calcula la entropía de la distribución de probabilidad."""
        probs = tf.nn.softmax(logits, axis=-1)
        # Evitar log(0)
        probs = tf.clip_by_value(probs, 1e-10, 1.0)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs), axis=-1)
        return float(tf.reduce_mean(entropy))
    
    def check_response_patterns(self, response: str) -> bool:
        """Verifica si la respuesta contiene patrones de incertidumbre."""
        return bool(self.uncertainty_regex.search(response))
    
    def verify_with_memory(self, question: str, response: str) -> Optional[bool]:
        """
        Verifica la respuesta contra el conocimiento almacenado.
        Returns: True si es correcta, False si incorrecta, None si no hay referencia.
        """
        if self.memory is None:
            return None
        
        # Buscar conocimiento relacionado
        related = self.memory.search_knowledge(question, limit=3)
        
        if not related:
            return None
        
        # Comparar respuesta con conocimiento almacenado
        for item in related:
            stored_answer = item.get("answer", "").lower()
            response_lower = response.lower()
            
            # Verificación simple: ¿la respuesta contiene la información correcta?
            if stored_answer in response_lower or response_lower in stored_answer:
                return True
        
        return False
    
    def detect(
        self,
        model,
        input_ids: tf.Tensor,
        generated_response: str = None,
        question: str = None
    ) -> Dict:
        """
        Detecta si el modelo sabe o no sabe responder.
        
        Returns:
            Dict con:
            - knows: bool - Si el modelo sabe
            - confidence: float - Nivel de confianza
            - entropy: float - Entropía de la salida
            - reasons: List[str] - Razones de la detección
        """
        reasons = []
        
        # 1. Obtener confianza del modelo
        logits, model_confidence = model(input_ids, return_confidence=True)
        confidence = float(model_confidence[0])
        
        if confidence < self.confidence_threshold:
            reasons.append(f"Confianza baja: {confidence:.2f} < {self.confidence_threshold}")
        
        # 2. Calcular entropía
        entropy = self.calculate_entropy(logits)
        
        if entropy > self.entropy_threshold:
            reasons.append(f"Entropía alta: {entropy:.2f} > {self.entropy_threshold}")
        
        # 3. Verificar patrones en respuesta
        if generated_response:
            if self.check_response_patterns(generated_response):
                reasons.append("Respuesta contiene patrones de incertidumbre")
        
        # 4. Verificar con memoria
        if question and generated_response:
            memory_check = self.verify_with_memory(question, generated_response)
            if memory_check is False:
                reasons.append("Respuesta contradice conocimiento almacenado")
            elif memory_check is True:
                # Bonus de confianza si coincide con memoria
                confidence = min(1.0, confidence + 0.2)
        
        # Decisión final
        knows = len(reasons) == 0
        
        return {
            "knows": knows,
            "confidence": confidence,
            "entropy": entropy,
            "reasons": reasons,
            "should_learn": not knows and confidence < 0.5
        }
    
    def quick_check(self, model, input_ids: tf.Tensor) -> Tuple[bool, float]:
        """Verificación rápida solo con confianza del modelo."""
        _, confidence = model(input_ids, return_confidence=True)
        conf_value = float(confidence[0])
        return conf_value >= self.confidence_threshold, conf_value


class LearningPrioritizer:
    """
    Prioriza qué debe aprender el modelo basándose en:
    - Frecuencia de preguntas no respondidas
    - Importancia del tema
    - Dependencias de conocimiento
    """
    
    def __init__(self):
        self.failed_questions = {}  # question -> count
        self.topic_importance = {}  # topic -> score
    
    def record_failure(self, question: str, topic: str = None):
        """Registra una pregunta que no se pudo responder."""
        self.failed_questions[question] = self.failed_questions.get(question, 0) + 1
        
        if topic:
            self.topic_importance[topic] = self.topic_importance.get(topic, 0) + 1
    
    def get_priority_topics(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Obtiene los temas más importantes para aprender."""
        sorted_topics = sorted(
            self.topic_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:top_n]
    
    def get_priority_questions(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Obtiene las preguntas más frecuentes sin responder."""
        sorted_questions = sorted(
            self.failed_questions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_questions[:top_n]
    
    def should_prioritize(self, question: str, threshold: int = 3) -> bool:
        """Determina si una pregunta debe priorizarse para aprendizaje."""
        return self.failed_questions.get(question, 0) >= threshold


if __name__ == "__main__":
    # Test del detector
    detector = IgnoranceDetector(confidence_threshold=0.7)
    
    # Test patrones
    test_responses = [
        "La capital de Francia es París",
        "No estoy seguro de la respuesta",
        "Creo que podría ser 42",
        "No sé qué es eso",
        "La respuesta es definitivamente 100",
    ]
    
    print("Test de patrones de incertidumbre:")
    for response in test_responses:
        has_uncertainty = detector.check_response_patterns(response)
        print(f"  '{response[:40]}...' -> Incertidumbre: {has_uncertainty}")
