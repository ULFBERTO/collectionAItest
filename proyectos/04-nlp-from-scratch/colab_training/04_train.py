# ============================================================
# SECCIÓN 4: ENTRENAMIENTO
# ============================================================

import tensorflow as tf
import os
import time
import json

class OxideTrainer:
    def __init__(self, model, dataset, output_path, learning_rate=1e-4):
        self.model = model
        self.dataset = dataset
        self.output_path = output_path
        
        # Optimizer con warmup
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        
        # Loss
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )
        
        # Métricas
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        
        # Historial
        self.history = {"loss": [], "accuracy": [], "epoch_time": []}
    
    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(targets, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(targets, predictions)
        
        return loss

    def train(self, epochs=10, save_every=2):
        """Entrena el modelo."""
        print(f"\n🚀 Iniciando entrenamiento por {epochs} épocas")
        print("=" * 50)
        
        for epoch in range(epochs):
            start_time = time.time()
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            
            # Progress bar manual
            num_batches = 0
            for batch, (inputs, targets) in enumerate(self.dataset):
                loss = self.train_step(inputs, targets)
                num_batches += 1
                
                if batch % 100 == 0:
                    print(f"  Batch {batch}: loss={loss:.4f}", end="\r")
            
            epoch_time = time.time() - start_time
            
            # Guardar métricas
            self.history["loss"].append(float(self.train_loss.result()))
            self.history["accuracy"].append(float(self.train_accuracy.result()))
            self.history["epoch_time"].append(epoch_time)
            
            print(f"Época {epoch+1}/{epochs} | "
                  f"Loss: {self.train_loss.result():.4f} | "
                  f"Acc: {self.train_accuracy.result():.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Guardar checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\n✅ Entrenamiento completado")
        self.save_final()
        return self.history
    
    def save_checkpoint(self, epoch):
        """Guarda checkpoint."""
        ckpt_path = os.path.join(self.output_path, f"ckpt_epoch_{epoch}")
        self.model.save_weights(ckpt_path)
        print(f"  💾 Checkpoint guardado: {ckpt_path}")
    
    def save_final(self):
        """Guarda modelo final y configuración."""
        # Guardar modelo completo
        model_path = os.path.join(self.output_path, "oxide_llm_final")
        self.model.save(model_path)
        
        # Guardar configuración
        config = self.model.get_config()
        config_path = os.path.join(self.output_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Guardar historial
        history_path = os.path.join(self.output_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✅ Modelo final guardado en: {model_path}")


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8):
    """Genera texto con el modelo."""
    tokens = tokenizer.encode(prompt)
    
    for _ in range(max_tokens):
        input_ids = tf.constant([tokens[-512:]], dtype=tf.int32)
        logits = model(input_ids, training=False)
        logits = logits[0, -1, :] / temperature
        
        # Top-k sampling
        top_k = 40
        top_logits, top_indices = tf.math.top_k(logits, k=top_k)
        probs = tf.nn.softmax(top_logits)
        next_token = tf.random.categorical(tf.expand_dims(tf.math.log(probs), 0), 1)
        next_token = top_indices[next_token[0, 0]].numpy()
        
        tokens.append(int(next_token))
        
        # Stop en EOS
        if next_token == 3:  # </s>
            break
    
    return tokenizer.decode(tokens)


# Uso:
# trainer = OxideTrainer(model, dataset, OUTPUT_PATH)
# history = trainer.train(epochs=20, save_every=5)
# 
# # Probar generación
# text = generate_text(model, loader.sp, "En un lugar de la Mancha")
# print(text)
