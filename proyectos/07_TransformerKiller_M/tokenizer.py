import os

class CharacterTokenizer:
    def __init__(self):
        self.chars = []
        self.vocab_size = 0
        self.stoi = {} 
        self.itos = {} 
        self.special_tokens = ["<|pad|>", "<|user|>", "<|assistant|>", "<|end|>"]

    def fit(self, text: str):
        chars = sorted(list(set(text)))
        self.chars = self.special_tokens + chars
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        print(f"Vocabulario SSM creado. Tamaño: {self.vocab_size}")

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l: list[int]) -> str:
        return ''.join([self.itos[i] for i in l])

def load_books(data_path: str) -> str:
    all_text = ""
    if not os.path.exists(data_path):
        return "Error: Carpeta de datos no encontrada."
    for filename in os.listdir(data_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                    all_text += f.read() + "\n"
            except:
                continue
    return all_text
