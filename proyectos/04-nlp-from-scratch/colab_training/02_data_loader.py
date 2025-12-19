# ============================================================
# SECCIÓN 2: DATA LOADER CON BPE TOKENIZER
# ============================================================

import os
import glob
import sentencepiece as spm
import tensorflow as tf
import numpy as np

class BPEDataLoader:
    def __init__(self, data_path, tokenizer_path, vocab_size=8000):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.sp = None
        
    def load_all_texts(self):
        """Carga todos los archivos .txt del directorio."""
        all_text = []
        txt_files = glob.glob(os.path.join(self.data_path, "*.txt"))
        
        print(f"📚 Encontrados {len(txt_files)} archivos")
        
        for filepath in txt_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                all_text.append(text)
                print(f"  ✓ {os.path.basename(filepath)}: {len(text):,} chars")
        
        combined = "\n\n".join(all_text)
        print(f"\n📊 Total: {len(combined):,} caracteres")
        return combined
    
    def train_tokenizer(self, text):
        """Entrena tokenizer BPE con SentencePiece."""
        # Guardar texto temporal para entrenar
        temp_file = "/content/temp_corpus.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        model_prefix = os.path.join(self.tokenizer_path, "oxide_bpe")
        
        # Entrenar SentencePiece
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
        )

        # Cargar tokenizer entrenado
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_prefix}.model")
        
        os.remove(temp_file)
        print(f"✅ Tokenizer entrenado: {self.vocab_size} tokens")
        return self.sp
    
    def load_tokenizer(self):
        """Carga tokenizer existente."""
        model_path = os.path.join(self.tokenizer_path, "oxide_bpe.model")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        print(f"✅ Tokenizer cargado: {self.sp.get_piece_size()} tokens")
        return self.sp
    
    def encode(self, text):
        """Codifica texto a tokens."""
        return self.sp.encode(text)
    
    def decode(self, tokens):
        """Decodifica tokens a texto."""
        return self.sp.decode(tokens)
    
    def create_dataset(self, text, seq_length=512, batch_size=32):
        """Crea dataset de TensorFlow."""
        # Tokenizar todo el texto
        tokens = self.encode(text)
        tokens = np.array(tokens, dtype=np.int32)
        
        print(f"📊 Tokens totales: {len(tokens):,}")
        
        # Crear secuencias
        def make_sequences():
            for i in range(0, len(tokens) - seq_length, seq_length // 2):
                chunk = tokens[i:i + seq_length + 1]
                if len(chunk) == seq_length + 1:
                    yield chunk[:-1], chunk[1:]
        
        dataset = tf.data.Dataset.from_generator(
            make_sequences,
            output_signature=(
                tf.TensorSpec(shape=(seq_length,), dtype=tf.int32),
                tf.TensorSpec(shape=(seq_length,), dtype=tf.int32)
            )
        )
        
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

# Uso:
# loader = BPEDataLoader(DATA_PATH, TOKENIZER_PATH, vocab_size=8000)
# text = loader.load_all_texts()
# loader.train_tokenizer(text)  # Solo la primera vez
# dataset = loader.create_dataset(text, seq_length=512, batch_size=32)
