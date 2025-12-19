#!/usr/bin/env python3
"""
Modo chat de OxideLearn.
Usa la memoria de conocimiento aprendido para responder.

Uso:
    python chat.py
"""

import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import KNOWLEDGE_DB_PATH


class SimpleChat:
    """Chat simple basado en memoria."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas de la memoria."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM algorithms")
        algorithms = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM rules")
        rules = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM facts")
        facts = cursor.fetchone()[0]
        
        return {"algorithms": algorithms, "rules": rules, "facts": facts}
    
    def list_algorithms(self) -> list:
        """Lista todos los algoritmos."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, description FROM algorithms")
        return cursor.fetchall()
    
    def get_algorithm(self, name: str) -> dict:
        """Busca un algoritmo por nombre."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name, description, steps, examples FROM algorithms WHERE name LIKE ?",
            (f"%{name}%",)
        )
        row = cursor.fetchone()
        if row:
            import json
            return {
                "name": row[0],
                "description": row[1],
                "steps": json.loads(row[2]) if row[2] else [],
                "examples": json.loads(row[3]) if row[3] else []
            }
        return None
    
    def search_facts(self, query: str) -> list:
        """Busca hechos relacionados."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT subject, predicate, object FROM facts 
               WHERE subject LIKE ? OR object LIKE ? OR predicate LIKE ?
               LIMIT 10""",
            (f"%{query}%", f"%{query}%", f"%{query}%")
        )
        return cursor.fetchall()
    
    def get_rules(self, category: str = None) -> list:
        """Obtiene reglas."""
        cursor = self.conn.cursor()
        if category:
            cursor.execute(
                "SELECT category, rule_text, examples FROM rules WHERE category LIKE ?",
                (f"%{category}%",)
            )
        else:
            cursor.execute("SELECT category, rule_text, examples FROM rules")
        return cursor.fetchall()
    
    def answer(self, question: str) -> str:
        """Intenta responder una pregunta."""
        q = question.lower().strip()
        q = q.replace("¿", "").replace("?", "").replace("!", "")
        
        # Comandos especiales
        if q in ["memoria", "stats", "estadísticas"]:
            stats = self.get_stats()
            return f"📊 Memoria:\n   Algoritmos: {stats['algorithms']}\n   Reglas: {stats['rules']}\n   Hechos: {stats['facts']}"
        
        if q in ["algoritmos", "listar algoritmos"]:
            algos = self.list_algorithms()
            if algos:
                return "📚 Algoritmos aprendidos:\n" + "\n".join(f"   • {a[0]}: {a[1][:50]}..." for a in algos)
            return "No hay algoritmos en memoria."
        
        if q in ["reglas", "listar reglas"]:
            rules = self.get_rules()
            if rules:
                return "📜 Reglas aprendidas:\n" + "\n".join(f"   • [{r[0]}] {r[1][:60]}..." for r in rules)
            return "No hay reglas en memoria."
        
        if q in ["hechos", "listar hechos"]:
            cursor = self.conn.cursor()
            cursor.execute("SELECT subject, predicate, object FROM facts LIMIT 20")
            facts = cursor.fetchall()
            if facts:
                return "📋 Hechos aprendidos:\n" + "\n".join(f"   • {f[0]} {f[1]} {f[2]}" for f in facts)
            return "No hay hechos en memoria."
        
        # Buscar algoritmos
        algo_keywords = ["cómo", "como", "qué es", "que es", "explica", "funciona", "algoritmo"]
        if any(kw in q for kw in algo_keywords):
            # Extraer tema
            words = q.split()
            for word in words:
                if len(word) > 3 and word not in ["cómo", "como", "funciona", "explica"]:
                    algo = self.get_algorithm(word)
                    if algo:
                        response = f"📖 {algo['name']}\n\n{algo['description']}\n\n"
                        if algo['steps']:
                            response += "Pasos:\n"
                            for i, step in enumerate(algo['steps'], 1):
                                response += f"   {i}. {step}\n"
                        if algo['examples']:
                            response += "\nEjemplos:\n"
                            for ex in algo['examples'][:2]:
                                if isinstance(ex, dict):
                                    response += f"   • {ex.get('input', '')} → {ex.get('output', '')}\n"
                        return response.strip()
        
        # Buscar hechos
        fact_keywords = ["capital", "cuál", "cual", "dónde", "donde", "quién", "quien", "cuánto", "cuanto"]
        if any(kw in q for kw in fact_keywords):
            words = q.split()
            for word in words:
                if len(word) > 3:
                    facts = self.search_facts(word)
                    if facts:
                        responses = []
                        for f in facts[:5]:
                            pred = f[1].replace("_", " ")
                            responses.append(f"{f[0]} {pred} {f[2]}")
                        return "\n".join(responses)
        
        # Buscar reglas
        if "regla" in q or "gramática" in q or "ortografía" in q:
            rules = self.get_rules()
            if rules:
                return f"📜 {rules[0][1]}"
        
        # Búsqueda general en algoritmos
        algos = self.list_algorithms()
        for algo_name, algo_desc in algos:
            if any(word in algo_name.lower() or word in algo_desc.lower() 
                   for word in q.split() if len(word) > 3):
                algo = self.get_algorithm(algo_name)
                if algo:
                    return f"📖 {algo['name']}: {algo['description']}"
        
        # Búsqueda general en hechos
        for word in q.split():
            if len(word) > 3:
                facts = self.search_facts(word)
                if facts:
                    f = facts[0]
                    return f"{f[0]} {f[1].replace('_', ' ')} {f[2]}"
        
        return "🤔 No encontré información sobre eso en mi memoria.\n   Usa 'algoritmos', 'reglas' o 'hechos' para ver qué sé."
    
    def close(self):
        self.conn.close()


def main():
    print("\n" + "=" * 50)
    print("🧠 OxideLearn - Chat")
    print("=" * 50)
    
    if not os.path.exists(KNOWLEDGE_DB_PATH):
        print("❌ No hay base de conocimiento. Ejecuta auto_learn.py primero.")
        return
    
    chat = SimpleChat(KNOWLEDGE_DB_PATH)
    stats = chat.get_stats()
    
    print(f"✅ Memoria cargada:")
    print(f"   Algoritmos: {stats['algorithms']}")
    print(f"   Reglas: {stats['rules']}")
    print(f"   Hechos: {stats['facts']}")
    
    print("\n💬 Comandos especiales:")
    print("   'algoritmos' - listar algoritmos")
    print("   'reglas' - listar reglas")
    print("   'hechos' - listar hechos")
    print("   'salir' - terminar")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 Tú: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("👋 ¡Hasta luego!")
                break
            
            response = chat.answer(user_input)
            print(f"\n🧠 Oxide: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
    
    chat.close()


if __name__ == "__main__":
    main()
