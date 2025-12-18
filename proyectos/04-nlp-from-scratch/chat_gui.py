import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
from generate import load_model, generate_text

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OxideLLM_5M GPT")
        self.root.geometry("600x700")
        
        # Cargar modelo en segundo plano
        self.model = None
        self.char2idx = None
        self.idx2char = None
        self.loading_label = tk.Label(root, text="Cargando modelo...", font=("Arial", 12))
        self.loading_label.pack(pady=20)
        
        threading.Thread(target=self.load_ai, daemon=True).start()

        # Area de chat
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)

        # Frame de entrada
        input_frame = tk.Frame(root)
        input_frame.pack(padx=10, pady=10, fill=tk.X)

        self.input_field = tk.Entry(input_frame, font=("Arial", 12))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(input_frame, text="Enviar", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

    def load_ai(self):
        self.model, self.char2idx, self.idx2char = load_model()
        self.root.after(0, self.on_model_loaded)

    def on_model_loaded(self):
        self.loading_label.destroy()
        if self.model:
            self.append_message("Sistema", "Modelo cargado correctamente. Escribe una frase para que OxideLLM_5M la continúe.")
        else:
            self.append_message("Error", "No se pudo cargar el modelo. Asegúrate de haber entrenado primero.")

    def send_message(self, event=None):
        user_input = self.input_field.get()
        if not user_input or not self.model:
            return
        
        self.input_field.delete(0, tk.END)
        self.append_message("Tú", user_input)
        
        # Generar respuesta en hilo separado para no congelar GUI
        threading.Thread(target=self.generate_response, args=(user_input,), daemon=True).start()

    def generate_response(self, prompt):
        try:
            # Generar
            generated = generate_text(self.model, prompt, self.char2idx, self.idx2char, num_generate=300)
            # Solo mostrar la parte nueva (opcional, aquí mostramos todo)
            response = generated
            self.root.after(0, lambda: self.append_message("OxideLLM_5M", response))
        except Exception as e:
            self.root.after(0, lambda: self.append_message("Error", str(e)))

    def append_message(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"\n[{sender}]:\n")
        self.chat_area.insert(tk.END, f"{message}\n")
        self.chat_area.insert(tk.END, "-"*50 + "\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
