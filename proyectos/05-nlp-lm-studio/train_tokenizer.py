from tokenizers import ByteLevelBPETokenizer
import os
import requests

def download_data(path='don_quijote.txt'):
    if not os.path.exists(path):
        url = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
        print(f"Descargando OxideLLM_5M de {url}...")
        response = requests.get(url)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    return path

def train_tokenizer():
    # 1. Preparar datos
    data_path = download_data()
    
    # 2. Inicializar tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # 3. Entrenar
    print("Entrenando tokenizer...")
    tokenizer.train(files=[data_path], vocab_size=5000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    
    # 4. Guardar
    save_dir = "./tokenizer_gpt2_quijote"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    tokenizer.save_model(save_dir)
    print(f"Tokenizer guardado en {save_dir}")

if __name__ == "__main__":
    train_tokenizer()
