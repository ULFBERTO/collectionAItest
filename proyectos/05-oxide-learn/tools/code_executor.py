"""
Ejecutor de código seguro para OxideLearn.
Permite ejecutar Python para cálculos matemáticos y lógicos.
"""

import ast
import sys
import math
import random
from io import StringIO
from typing import Tuple, Optional
import re


class SafeCodeExecutor:
    """
    Ejecuta código Python de forma segura y limitada.
    Solo permite operaciones matemáticas y lógicas básicas.
    """
    
    ALLOWED_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'chr', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'hex', 'int',
        'len', 'list', 'map', 'max', 'min', 'oct', 'ord', 'pow',
        'print', 'range', 'reversed', 'round', 'set', 'sorted',
        'str', 'sum', 'tuple', 'zip', 'True', 'False', 'None'
    }
    
    ALLOWED_MODULES = {
        'math': math,
        'random': random,
    }
    
    # Nodos AST peligrosos
    FORBIDDEN_NODES = {
        ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
        ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
        ast.Await, ast.Yield, ast.YieldFrom
    }
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.max_iterations = 10000
    
    def _check_ast(self, tree: ast.AST) -> Tuple[bool, str]:
        """Verifica que el AST no contenga operaciones peligrosas."""
        for node in ast.walk(tree):
            # Verificar nodos prohibidos
            if type(node) in self.FORBIDDEN_NODES:
                return False, f"Operación no permitida: {type(node).__name__}"
            
            # Verificar llamadas a funciones
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in self.ALLOWED_BUILTINS:
                        # Podría ser una función definida en el código
                        pass
                elif isinstance(node.func, ast.Attribute):
                    # Verificar acceso a atributos peligrosos
                    if node.func.attr.startswith('_'):
                        return False, f"Acceso a atributo privado no permitido"
            
            # Verificar acceso a atributos
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    return False, f"Acceso a atributo privado no permitido"
        
        return True, ""
    
    def _create_safe_globals(self) -> dict:
        """Crea un entorno global seguro."""
        safe_builtins = {
            name: getattr(__builtins__, name) if hasattr(__builtins__, name) 
            else __builtins__[name] if isinstance(__builtins__, dict) else None
            for name in self.ALLOWED_BUILTINS
        }
        safe_builtins = {k: v for k, v in safe_builtins.items() if v is not None}
        
        # Agregar print personalizado
        output_buffer = []
        def safe_print(*args, **kwargs):
            output_buffer.append(' '.join(str(a) for a in args))
        safe_builtins['print'] = safe_print
        
        safe_globals = {
            '__builtins__': safe_builtins,
            '__output__': output_buffer,
        }
        
        # Agregar módulos permitidos
        for name, module in self.ALLOWED_MODULES.items():
            safe_globals[name] = module
        
        return safe_globals
    
    def execute(self, code: str) -> Tuple[bool, str, Optional[str]]:
        """
        Ejecuta código de forma segura.
        
        Returns:
            Tuple[success, output/error, result]
        """
        # Limpiar código
        code = code.strip()
        if not code:
            return False, "Código vacío", None
        
        # Parsear AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Error de sintaxis: {e}", None
        
        # Verificar seguridad
        is_safe, error = self._check_ast(tree)
        if not is_safe:
            return False, error, None
        
        # Crear entorno seguro
        safe_globals = self._create_safe_globals()
        safe_locals = {}
        
        # Capturar stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # Ejecutar
            exec(compile(tree, '<string>', 'exec'), safe_globals, safe_locals)
            
            # Obtener output
            stdout_output = sys.stdout.getvalue()
            print_output = '\n'.join(safe_globals['__output__'])
            
            output = print_output or stdout_output
            
            # Buscar resultado (última expresión o variable 'resultado')
            result = None
            if 'resultado' in safe_locals:
                result = str(safe_locals['resultado'])
            elif 'result' in safe_locals:
                result = str(safe_locals['result'])
            elif 'respuesta' in safe_locals:
                result = str(safe_locals['respuesta'])
            
            return True, output.strip(), result
            
        except Exception as e:
            return False, f"Error de ejecución: {type(e).__name__}: {e}", None
        finally:
            sys.stdout = old_stdout
    
    def solve_math(self, expression: str) -> Tuple[bool, str]:
        """
        Resuelve una expresión matemática.
        
        Args:
            expression: Expresión como "2 + 2" o "sqrt(16)"
        
        Returns:
            Tuple[success, result]
        """
        # Limpiar expresión
        expr = expression.strip()
        
        # Reemplazar funciones comunes
        replacements = {
            'raíz': 'math.sqrt',
            'raiz': 'math.sqrt',
            'sqrt': 'math.sqrt',
            'sen': 'math.sin',
            'cos': 'math.cos',
            'tan': 'math.tan',
            'log': 'math.log',
            'ln': 'math.log',
            'exp': 'math.exp',
            'abs': 'abs',
            'pi': 'math.pi',
            'π': 'math.pi',
            '^': '**',
            '×': '*',
            '÷': '/',
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        code = f"resultado = {expr}\nprint(resultado)"
        success, output, result = self.execute(code)
        
        if success:
            return True, result or output
        return False, output


def extract_code_from_response(response: str) -> Optional[str]:
    """Extrae código Python de una respuesta."""
    # Buscar bloques de código
    code_patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'`([^`]+)`',
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    return None


if __name__ == "__main__":
    executor = SafeCodeExecutor()
    
    # Tests
    tests = [
        "2 + 2",
        "math.sqrt(16)",
        "sum(range(10))",
        "[x**2 for x in range(5)]",
        "resultado = 15 * 7\nprint(f'El resultado es {resultado}')",
    ]
    
    print("🧪 Probando ejecutor de código:\n")
    
    for test in tests:
        print(f"Código: {test}")
        success, output, result = executor.execute(f"resultado = {test}\nprint(resultado)")
        print(f"  Éxito: {success}")
        print(f"  Output: {output}")
        print(f"  Resultado: {result}")
        print()
    
    # Test solve_math
    print("\n🔢 Probando solve_math:\n")
    math_tests = ["2 + 2", "sqrt(144)", "3^4", "100 / 4"]
    
    for expr in math_tests:
        success, result = executor.solve_math(expr)
        print(f"  {expr} = {result} (éxito: {success})")
