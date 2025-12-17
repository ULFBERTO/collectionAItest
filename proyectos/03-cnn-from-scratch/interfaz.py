import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
import os

# Configuración para evitar errores de protobuf
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Cargar el modelo
print("Cargando modelo...")
try:
    modelo = tf.keras.models.load_model('modelo_ropa.h5')
    print("Modelo cargado.")
except:
    print("No se encontró 'modelo_ropa.h5'. Ejecuta main.py primero.")
    modelo = None

nombres_clases = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocesar_imagen(ruta_imagen):
    # 1. Abrir imagen
    img = Image.open(ruta_imagen)
    
    # 2. Convertir a escala de grises (L)
    img = img.convert('L')
    
    # 3. Invertir colores (Si el fondo es blanco y la ropa oscura)
    # Fashion MNIST es fondo negro y ropa blanca.
    # Asumimos que el usuario sube fotos normales (fondo claro).
    img = ImageOps.invert(img)
    
    # 4. Redimensionar a 28x28
    img = img.resize((28, 28))
    
    # 5. Convertir a array numpy y normalizar
    img_array = np.array(img) / 255.0
    
    # 6. Reshape para el modelo (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, img

def cargar_y_predecir():
    ruta_archivo = filedialog.askopenfilename()
    if not ruta_archivo:
        return
        
    # Preprocesar
    img_array, img_pil = preprocesar_imagen(ruta_archivo)
    
    # Predecir
    if modelo:
        prediccion = modelo.predict(img_array)
        indice = np.argmax(prediccion[0])
        clase = nombres_clases[indice]
        confianza = prediccion[0][indice] * 100
        
        resultado_texto.set(f"Predicción: {clase}\nConfianza: {confianza:.2f}%")
    else:
        resultado_texto.set("Error: Modelo no cargado")

    # Mostrar imagen en la GUI (redimensionada para verla mejor)
    img_mostrar = img_pil.resize((150, 150), Image.Resampling.NEAREST)
    img_tk = ImageTk.PhotoImage(img_mostrar)
    label_imagen.config(image=img_tk)
    label_imagen.image = img_tk

# Configuración de la Ventana
ventana = tk.Tk()
ventana.title("Clasificador de Ropa - Fashion MNIST")
ventana.geometry("300x350")

btn_cargar = tk.Button(ventana, text="Cargar Imagen", command=cargar_y_predecir)
btn_cargar.pack(pady=20)

label_imagen = tk.Label(ventana)
label_imagen.pack()

resultado_texto = tk.StringVar()
resultado_texto.set("Sube una imagen para probar")
label_resultado = tk.Label(ventana, textvariable=resultado_texto, font=("Helvetica", 12))
label_resultado.pack(pady=20)

ventana.mainloop()
