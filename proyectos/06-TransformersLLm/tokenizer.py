import os

class CharacterTokenizer:
    """
    Un tokenizador simple a nivel de caracteres.
    Ideal para entender la transformación de datos desde cero.
    Convierte cada letra/símbolo en un número único.
    """
    def __init__(self):
        self.chars = []
        self.vocab_size = 0
        self.stoi = {} # string to index
        self.itos = {} # index to string
        
        # Tokens especiales para estructura de chat
        self.special_tokens = ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"]

    def fit(self, text: str):
        """
        Crea el vocabulario basándose en el texto proporcionado.
        """
        # Extraer caracteres únicos y ordenarlos
        chars = sorted(list(set(text)))
        
        # Combinar tokens especiales con los caracteres encontrados
        self.chars = self.special_tokens + chars
        self.vocab_size = len(self.chars)
        
        # Crear diccionarios de mapeo
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        print(f"Vocabulario creado. Tamaño: {self.vocab_size} caracteres.")

    def encode(self, s: str) -> list[int]:
        """
        Convierte una cadena de texto en una lista de números.
        """
        # Si el caracter no está en el vocabulario, podríamos ignorarlo 
        # o usar un token <|unk|>, aquí simplemente lo ignoramos por simplicidad.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l: list[int]) -> str:
        """
        Convierte una lista de números de vuelta a texto.
        """
        return ''.join([self.itos[i] for i in l])

def load_books(data_path: str) -> str:
    """
    Lee todos los archivos .txt de la carpeta de libros y los concatena.
    """
    all_text = ""
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
    return all_text

if __name__ == "__main__":
    # Prueba del tokenizador
    path = r"C:\EVIROMENT\M\collectionAItest\proyectos\Data\libros_espanol"
    print("Cargando libros...")
    text = load_books(path)
    print(f"Total de caracteres cargados: {len(text)}")
    
    tokenizer = CharacterTokenizer()
    tokenizer.fit(text)
    
    prueba = "Hola, ¿cómo estás?"
    encoded = tokenizer.encode(prueba)
    decoded = tokenizer.decode(encoded)
    
    print(f"Texto original: {prueba}")
    print(f"Tokens: {encoded}")
    print(f"Decodificado: {decoded}")
