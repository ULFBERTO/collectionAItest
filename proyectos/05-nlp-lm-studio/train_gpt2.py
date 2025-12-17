from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TextDataset
import os

def train_gpt2():
    # 1. Cargar Tokenizer
    tokenizer_path = "./tokenizer_gpt2_quijote"
    if not os.path.exists(tokenizer_path):
        print("Error: Entrena el tokenizer primero.")
        return

    # Cargar como GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })

    # 2. Configuración del Modelo (GPT-2 Small)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=6,    # Reducido para entrenar rápido (GPT-2 original tiene 12)
        n_head=6,
    )

    model = GPT2LMHeadModel(config)
    
    # 3. Preparar Dataset
    print("Preparando dataset...")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="don_quijote.txt",
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # 4. Entrenamiento
    training_args = TrainingArguments(
        output_dir="./gpt2_quijote_checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Iniciando entrenamiento...")
    trainer.train()
    
    # 5. Guardar Modelo Final
    print("Guardando modelo final...")
    model.save_pretrained("./gpt2_quijote_model")
    tokenizer.save_pretrained("./gpt2_quijote_model")
    print("¡Modelo guardado en ./gpt2_quijote_model!")

if __name__ == "__main__":
    train_gpt2()
