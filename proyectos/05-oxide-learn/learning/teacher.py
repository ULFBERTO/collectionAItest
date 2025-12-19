"""
Conexión con el modelo maestro (Llama) para aprendizaje.
Soporta LM Studio, Ollama, y APIs compatibles con OpenAI.
"""

import requests
import json
from typing import Optional, Dict, List, Generator
import re


class TeacherLLM:
    """
    Interfaz con el modelo maestro (Llama u otro LLM grande).
    El maestro enseña al modelo pequeño cuando este no sabe algo.
    """
    
    def __init__(
        self,
        api_url="http://localhost:11434/v1",
        model_name="llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        self.api_url = api_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Verificar conexión
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Verifica que el servidor esté disponible."""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            if response.status_code == 200:
                print(f"✅ Conectado a maestro en {self.api_url}")
                return True
        except requests.exceptions.ConnectionError:
            print(f"⚠️ No se pudo conectar a {self.api_url}")
            print("   Asegúrate de que LM Studio u Ollama esté corriendo.")
        return False
    
    def ask(
        self,
        question: str,
        system_prompt: str = None,
        context: str = None
    ) -> Optional[str]:
        """
        Hace una pregunta al modelo maestro.
        
        Args:
            question: La pregunta a hacer
            system_prompt: Instrucciones del sistema
            context: Contexto adicional
        
        Returns:
            Respuesta del maestro o None si falla
        """
        if system_prompt is None:
            system_prompt = """Eres un maestro experto que enseña de forma clara y estructurada.
Cuando expliques algo:
1. Da la respuesta directa primero
2. Explica el razonamiento o algoritmo paso a paso
3. Proporciona ejemplos si es útil
4. Identifica el tipo de conocimiento (algoritmo, regla, hecho, razonamiento)

Responde en español de forma concisa pero completa."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            messages.append({"role": "user", "content": f"Contexto: {context}"})
        
        messages.append({"role": "user", "content": question})
        
        max_retries = 50
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens
                    },
                    timeout=300  # 5 minutos de timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f"❌ Error del servidor: {response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"   Reintentando ({attempt + 2}/{max_retries})...")
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                print(f"⏱️ Timeout (intento {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print("   Reintentando...")
                    continue
                print("❌ Timeout después de todos los reintentos")
                return None
            except Exception as e:
                print(f"❌ Error: {e}")
                if attempt < max_retries - 1:
                    print(f"   Reintentando ({attempt + 2}/{max_retries})...")
                    continue
                return None
        
        return None
    
    def teach_algorithm(self, topic: str) -> Optional[Dict]:
        """
        Pide al maestro que enseñe un algoritmo.
        
        Returns:
            Dict con: name, description, steps, examples
        """
        prompt = f"""Enséñame el algoritmo para: {topic}

Responde en este formato JSON exacto:
{{
    "name": "nombre del algoritmo",
    "description": "descripción breve",
    "steps": ["paso 1", "paso 2", "paso 3"],
    "examples": [
        {{"input": "ejemplo entrada", "output": "resultado", "explanation": "explicación"}}
    ]
}}

Solo responde con el JSON, sin texto adicional."""
        
        response = self.ask(prompt)
        if response:
            try:
                # Extraer JSON de la respuesta
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                print(f"⚠️ No se pudo parsear respuesta como JSON")
        return None
    
    def teach_rule(self, topic: str, category: str = "general") -> Optional[Dict]:
        """
        Pide al maestro que enseñe una regla.
        
        Returns:
            Dict con: rule_text, exceptions, examples
        """
        prompt = f"""Enséñame la regla sobre: {topic}
Categoría: {category}

Responde en este formato JSON exacto:
{{
    "rule_text": "la regla explicada claramente",
    "exceptions": ["excepción 1", "excepción 2"],
    "examples": ["ejemplo 1", "ejemplo 2", "ejemplo 3"]
}}

Solo responde con el JSON, sin texto adicional."""
        
        response = self.ask(prompt)
        if response:
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    def teach_facts(self, topic: str) -> Optional[List[Dict]]:
        """
        Pide al maestro hechos sobre un tema.
        
        Returns:
            Lista de Dict con: subject, predicate, object
        """
        prompt = f"""Dame los hechos más importantes sobre: {topic}

Responde en este formato JSON exacto (lista de hechos):
[
    {{"subject": "sujeto", "predicate": "relación", "object": "objeto"}},
    {{"subject": "sujeto2", "predicate": "relación2", "object": "objeto2"}}
]

Por ejemplo para "capitales de Europa":
[
    {{"subject": "Francia", "predicate": "tiene_capital", "object": "París"}},
    {{"subject": "España", "predicate": "tiene_capital", "object": "Madrid"}}
]

Solo responde con el JSON, sin texto adicional."""
        
        response = self.ask(prompt)
        if response:
            try:
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    def explain_reasoning(self, question: str) -> Optional[Dict]:
        """
        Pide al maestro que explique el razonamiento para resolver algo.
        
        Returns:
            Dict con: answer, reasoning_steps, conclusion
        """
        prompt = f"""Resuelve esto explicando tu razonamiento paso a paso: {question}

Responde en este formato JSON exacto:
{{
    "answer": "respuesta final",
    "reasoning_steps": [
        "Paso 1: ...",
        "Paso 2: ...",
        "Paso 3: ..."
    ],
    "conclusion": "por qué esta es la respuesta correcta"
}}

Solo responde con el JSON, sin texto adicional."""
        
        response = self.ask(prompt)
        if response:
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    def generate_training_examples(
        self,
        topic: str,
        num_examples: int = 10
    ) -> Optional[List[Dict]]:
        """
        Genera ejemplos de entrenamiento sobre un tema.
        
        Returns:
            Lista de Dict con: input, output
        """
        prompt = f"""Genera {num_examples} ejemplos de entrenamiento sobre: {topic}

Cada ejemplo debe tener una pregunta/entrada y su respuesta correcta.

Responde en este formato JSON exacto:
[
    {{"input": "pregunta o entrada", "output": "respuesta correcta"}},
    {{"input": "otra pregunta", "output": "otra respuesta"}}
]

Solo responde con el JSON, sin texto adicional."""
        
        response = self.ask(prompt)
        if response:
            try:
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    def verify_answer(self, question: str, answer: str) -> Optional[Dict]:
        """
        Verifica si una respuesta es correcta.
        
        Returns:
            Dict con: is_correct, correct_answer, explanation
        """
        prompt = f"""Verifica si esta respuesta es correcta:

Pregunta: {question}
Respuesta dada: {answer}

Responde en este formato JSON exacto:
{{
    "is_correct": true o false,
    "correct_answer": "la respuesta correcta",
    "explanation": "por qué es correcta o incorrecta"
}}

Solo responde con el JSON, sin texto adicional."""
        
        response = self.ask(prompt)
        if response:
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    # Normalizar is_correct a bool
                    if isinstance(result.get("is_correct"), str):
                        result["is_correct"] = result["is_correct"].lower() == "true"
                    return result
            except json.JSONDecodeError:
                pass
        return None


class CurriculumGenerator:
    """
    Genera currículum de aprendizaje usando el maestro.
    """
    
    def __init__(self, teacher: TeacherLLM):
        self.teacher = teacher
    
    def generate_curriculum(self, topic: str, depth: str = "básico") -> Optional[List[str]]:
        """
        Genera un currículum de subtemas para aprender.
        
        Args:
            topic: Tema principal
            depth: "básico", "intermedio", "avanzado"
        
        Returns:
            Lista de subtemas ordenados por dificultad
        """
        prompt = f"""Genera un currículum de aprendizaje para: {topic}
Nivel: {depth}

Lista los subtemas en orden de aprendizaje (de más básico a más avanzado).
Cada subtema debe ser específico y aprendible.

Responde en este formato JSON exacto:
{{
    "topic": "{topic}",
    "level": "{depth}",
    "subtopics": [
        "subtema 1 (más básico)",
        "subtema 2",
        "subtema 3",
        "subtema N (más avanzado)"
    ],
    "prerequisites": ["conocimiento previo necesario"]
}}

Solo responde con el JSON, sin texto adicional."""
        
        response = self.teacher.ask(prompt)
        if response:
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("subtopics", [])
            except json.JSONDecodeError:
                pass
        return None
    
    def generate_exercises(self, topic: str, num_exercises: int = 5) -> Optional[List[Dict]]:
        """
        Genera ejercicios de práctica.
        
        Returns:
            Lista de ejercicios con pregunta, respuesta, dificultad
        """
        prompt = f"""Genera {num_exercises} ejercicios de práctica sobre: {topic}

Varía la dificultad de fácil a difícil.

Responde en este formato JSON exacto:
[
    {{
        "question": "pregunta del ejercicio",
        "answer": "respuesta correcta",
        "difficulty": "fácil/medio/difícil",
        "hint": "pista opcional"
    }}
]

Solo responde con el JSON, sin texto adicional."""
        
        response = self.teacher.ask(prompt)
        if response:
            try:
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None


if __name__ == "__main__":
    # Test
    from config import TEACHER_CONFIG
    
    print("Probando conexión con el maestro...")
    teacher = TeacherLLM(**TEACHER_CONFIG)
    
    # Test pregunta simple
    print("\n--- Test: Pregunta simple ---")
    response = teacher.ask("¿Cuál es la capital de Francia?")
    if response:
        print(f"Respuesta: {response[:200]}...")
    
    # Test algoritmo
    print("\n--- Test: Enseñar algoritmo ---")
    algo = teacher.teach_algorithm("multiplicación de dos números")
    if algo:
        print(f"Algoritmo: {json.dumps(algo, indent=2, ensure_ascii=False)[:500]}...")
    
    # Test hechos
    print("\n--- Test: Enseñar hechos ---")
    facts = teacher.teach_facts("planetas del sistema solar")
    if facts:
        print(f"Hechos: {json.dumps(facts[:3], indent=2, ensure_ascii=False)}...")
