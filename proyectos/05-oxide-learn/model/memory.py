"""
Sistema de memoria persistente para OxideLearn.
Almacena conocimiento aprendido en diferentes categorías.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np


class KnowledgeMemory:
    """
    Base de datos de conocimiento aprendido.
    
    Tipos de conocimiento:
    - algorithm: Procedimientos paso a paso
    - rule: Reglas y patrones
    - fact: Información factual
    - reasoning: Cadenas de razonamiento
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Inicializa la base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla principal de conocimiento
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                explanation TEXT,
                confidence REAL DEFAULT 1.0,
                times_used INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'teacher'
            )
        """)
        
        # Tabla de algoritmos (procedimientos)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS algorithms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                steps TEXT NOT NULL,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de reglas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                rule_text TEXT NOT NULL,
                exceptions TEXT,
                examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla de hechos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Índices para búsqueda rápida
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        
        conn.commit()
        conn.close()
    
    def store_knowledge(
        self,
        knowledge_type: str,
        topic: str,
        question: str,
        answer: str,
        explanation: str = None,
        confidence: float = 1.0
    ) -> int:
        """Almacena nuevo conocimiento."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO knowledge (type, topic, question, answer, explanation, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (knowledge_type, topic, question, answer, explanation, confidence))
        
        knowledge_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return knowledge_id
    
    def store_algorithm(
        self,
        name: str,
        description: str,
        steps: List[str],
        examples: List[Dict] = None
    ) -> int:
        """Almacena un algoritmo/procedimiento."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        steps_json = json.dumps(steps, ensure_ascii=False)
        examples_json = json.dumps(examples or [], ensure_ascii=False)
        
        cursor.execute("""
            INSERT OR REPLACE INTO algorithms (name, description, steps, examples)
            VALUES (?, ?, ?, ?)
        """, (name, description, steps_json, examples_json))
        
        algo_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return algo_id
    
    def store_rule(
        self,
        category: str,
        rule_text: str,
        exceptions: List[str] = None,
        examples: List[str] = None
    ) -> int:
        """Almacena una regla."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        exceptions_json = json.dumps(exceptions or [], ensure_ascii=False)
        examples_json = json.dumps(examples or [], ensure_ascii=False)
        
        cursor.execute("""
            INSERT INTO rules (category, rule_text, exceptions, examples)
            VALUES (?, ?, ?, ?)
        """, (category, rule_text, exceptions_json, examples_json))
        
        rule_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return rule_id
    
    def store_fact(
        self,
        category: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0
    ) -> int:
        """Almacena un hecho (triple RDF-like)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO facts (category, subject, predicate, object, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (category, subject, predicate, obj, confidence))
        
        fact_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return fact_id
    
    def search_knowledge(
        self,
        query: str,
        knowledge_type: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """Busca conocimiento relevante."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if knowledge_type:
            cursor.execute("""
                SELECT * FROM knowledge 
                WHERE type = ? AND (topic LIKE ? OR question LIKE ? OR answer LIKE ?)
                ORDER BY confidence DESC, times_correct DESC
                LIMIT ?
            """, (knowledge_type, f"%{query}%", f"%{query}%", f"%{query}%", limit))
        else:
            cursor.execute("""
                SELECT * FROM knowledge 
                WHERE topic LIKE ? OR question LIKE ? OR answer LIKE ?
                ORDER BY confidence DESC, times_correct DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_algorithm(self, name: str) -> Optional[Dict]:
        """Obtiene un algoritmo por nombre."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM algorithms WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            columns = ["id", "name", "description", "steps", "examples", "created_at"]
            result = dict(zip(columns, row))
            result["steps"] = json.loads(result["steps"])
            result["examples"] = json.loads(result["examples"])
            return result
        
        return None
    
    def get_facts_about(self, subject: str) -> List[Dict]:
        """Obtiene todos los hechos sobre un sujeto."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM facts WHERE subject LIKE ?
            ORDER BY confidence DESC
        """, (f"%{subject}%",))
        
        columns = ["id", "category", "subject", "predicate", "object", "confidence", "created_at"]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_rules_by_category(self, category: str) -> List[Dict]:
        """Obtiene reglas de una categoría."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM rules WHERE category = ?", (category,))
        
        columns = ["id", "category", "rule_text", "exceptions", "examples", "created_at"]
        results = []
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            result["exceptions"] = json.loads(result["exceptions"])
            result["examples"] = json.loads(result["examples"])
            results.append(result)
        
        conn.close()
        return results
    
    def update_usage(self, knowledge_id: int, was_correct: bool):
        """Actualiza estadísticas de uso."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if was_correct:
            cursor.execute("""
                UPDATE knowledge 
                SET times_used = times_used + 1, 
                    times_correct = times_correct + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (knowledge_id,))
        else:
            cursor.execute("""
                UPDATE knowledge 
                SET times_used = times_used + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (knowledge_id,))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la memoria."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total conocimiento
        cursor.execute("SELECT COUNT(*) FROM knowledge")
        stats["total_knowledge"] = cursor.fetchone()[0]
        
        # Por tipo
        cursor.execute("SELECT type, COUNT(*) FROM knowledge GROUP BY type")
        stats["by_type"] = dict(cursor.fetchall())
        
        # Algoritmos
        cursor.execute("SELECT COUNT(*) FROM algorithms")
        stats["algorithms"] = cursor.fetchone()[0]
        
        # Reglas
        cursor.execute("SELECT COUNT(*) FROM rules")
        stats["rules"] = cursor.fetchone()[0]
        
        # Hechos
        cursor.execute("SELECT COUNT(*) FROM facts")
        stats["facts"] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def export_to_json(self, filepath: str):
        """Exporta toda la memoria a JSON."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = {
            "knowledge": [],
            "algorithms": [],
            "rules": [],
            "facts": []
        }
        
        # Knowledge
        cursor.execute("SELECT * FROM knowledge")
        columns = [desc[0] for desc in cursor.description]
        data["knowledge"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Algorithms
        cursor.execute("SELECT * FROM algorithms")
        for row in cursor.fetchall():
            algo = {
                "id": row[0], "name": row[1], "description": row[2],
                "steps": json.loads(row[3]), "examples": json.loads(row[4]),
                "created_at": row[5]
            }
            data["algorithms"].append(algo)
        
        # Rules
        cursor.execute("SELECT * FROM rules")
        for row in cursor.fetchall():
            rule = {
                "id": row[0], "category": row[1], "rule_text": row[2],
                "exceptions": json.loads(row[3]), "examples": json.loads(row[4]),
                "created_at": row[5]
            }
            data["rules"].append(rule)
        
        # Facts
        cursor.execute("SELECT * FROM facts")
        columns = ["id", "category", "subject", "predicate", "object", "confidence", "created_at"]
        data["facts"] = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ Memoria exportada a {filepath}")


if __name__ == "__main__":
    # Test
    import tempfile
    
    db_path = tempfile.mktemp(suffix=".db")
    memory = KnowledgeMemory(db_path)
    
    # Almacenar conocimiento
    memory.store_knowledge(
        "algorithm", "matemáticas", 
        "¿Cómo sumar dos números?",
        "Combinar las cantidades",
        "La suma es la operación de añadir cantidades"
    )
    
    memory.store_algorithm(
        "multiplicación",
        "Multiplicar dos números",
        ["Tomar el primer número", "Sumarlo consigo mismo N veces", "N es el segundo número"],
        [{"input": "3 × 4", "output": "12"}]
    )
    
    memory.store_fact("geografía", "Francia", "tiene_capital", "París")
    memory.store_fact("geografía", "España", "tiene_capital", "Madrid")
    
    memory.store_rule(
        "gramática",
        "Los sustantivos terminados en -ción son femeninos",
        ["corazón (masculino)"],
        ["la canción", "la nación"]
    )
    
    # Buscar
    print("Búsqueda 'suma':", memory.search_knowledge("suma"))
    print("\nAlgoritmo 'multiplicación':", memory.get_algorithm("multiplicación"))
    print("\nHechos sobre 'Francia':", memory.get_facts_about("Francia"))
    print("\nReglas de 'gramática':", memory.get_rules_by_category("gramática"))
    print("\nEstadísticas:", memory.get_stats())
    
    # Limpiar
    os.remove(db_path)
