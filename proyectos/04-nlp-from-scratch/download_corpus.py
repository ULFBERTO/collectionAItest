"""
Descarga un corpus de libros clásicos en español desde Project Gutenberg.
Los guarda en la carpeta Data para usar en entrenamiento.
"""

import os
import requests
import time

# Carpeta de destino
DATA_DIR = "../Data/libros_espanol"

# Lista de libros en español de Project Gutenberg
# Formato: (ID, nombre_archivo, descripcion)
LIBROS = [
    (2000, "don_quijote_1.txt", "Don Quijote - Parte 1 - Cervantes"),
    (2001, "don_quijote_2.txt", "Don Quijote - Parte 2 - Cervantes"),
    (17073, "la_celestina.txt", "La Celestina - Fernando de Rojas"),
    (15532, "lazarillo_de_tormes.txt", "Lazarillo de Tormes - Anónimo"),
    (24536, "el_buscón.txt", "El Buscón - Quevedo"),
    (49836, "novelas_ejemplares.txt", "Novelas Ejemplares - Cervantes"),
    (14765, "la_gitanilla.txt", "La Gitanilla - Cervantes"),
    (60829, "rinconete_cortadillo.txt", "Rinconete y Cortadillo - Cervantes"),
    (15353, "el_licenciado_vidriera.txt", "El Licenciado Vidriera - Cervantes"),
    (21532, "la_ilustre_fregona.txt", "La Ilustre Fregona - Cervantes"),
    (36674, "el_coloquio_perros.txt", "El Coloquio de los Perros - Cervantes"),
    (44087, "la_fuerza_sangre.txt", "La Fuerza de la Sangre - Cervantes"),
    (15027, "el_celoso_extremeno.txt", "El Celoso Extremeño - Cervantes"),
    (60830, "las_dos_doncellas.txt", "Las Dos Doncellas - Cervantes"),
    (60831, "la_senora_cornelia.txt", "La Señora Cornelia - Cervantes"),
    (60832, "el_casamiento_enganoso.txt", "El Casamiento Engañoso - Cervantes"),
    (4363, "poesias_garcilaso.txt", "Poesías - Garcilaso de la Vega"),
    (10676, "rimas_becquer.txt", "Rimas - Gustavo Adolfo Bécquer"),
    (17898, "leyendas_becquer.txt", "Leyendas - Gustavo Adolfo Bécquer"),
    (49010, "pepita_jimenez.txt", "Pepita Jiménez - Juan Valera"),
    (36756, "dona_perfecta.txt", "Doña Perfecta - Benito Pérez Galdós"),
    (39131, "marianela.txt", "Marianela - Benito Pérez Galdós"),
    (16625, "la_regenta_1.txt", "La Regenta Parte 1 - Leopoldo Alas Clarín"),
    (16626, "la_regenta_2.txt", "La Regenta Parte 2 - Leopoldo Alas Clarín"),
    (49019, "fortunata_jacinta_1.txt", "Fortunata y Jacinta 1 - Galdós"),
    (49020, "fortunata_jacinta_2.txt", "Fortunata y Jacinta 2 - Galdós"),
    (49021, "fortunata_jacinta_3.txt", "Fortunata y Jacinta 3 - Galdós"),
    (49022, "fortunata_jacinta_4.txt", "Fortunata y Jacinta 4 - Galdós"),
]


def download_book(book_id: int, filename: str, description: str) -> bool:
    """Descarga un libro de Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    filepath = os.path.join(DATA_DIR, filename)
    
    # Si ya existe, saltar
    if os.path.exists(filepath):
        print(f"  ✓ Ya existe: {filename}")
        return True
    
    try:
        print(f"  ⬇️ Descargando: {description}...")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Guardar archivo
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"  ✅ Guardado: {filename} ({len(response.text):,} caracteres)")
            return True
        else:
            print(f"  ❌ Error {response.status_code}: {filename}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error descargando {filename}: {e}")
        return False


def download_all():
    """Descarga todos los libros del corpus."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"📚 Descargando corpus de {len(LIBROS)} libros en español...")
    print(f"📁 Destino: {os.path.abspath(DATA_DIR)}\n")
    
    successful = 0
    failed = 0
    
    for book_id, filename, description in LIBROS:
        if download_book(book_id, filename, description):
            successful += 1
        else:
            failed += 1
        # Pequeña pausa para no sobrecargar el servidor
        time.sleep(0.5)
    
    print(f"\n{'='*50}")
    print(f"✅ Descargados: {successful}")
    print(f"❌ Fallidos: {failed}")
    print(f"📁 Ubicación: {os.path.abspath(DATA_DIR)}")
    
    # Calcular tamaño total
    total_size = 0
    total_chars = 0
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isfile(filepath):
            total_size += os.path.getsize(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                total_chars += len(f.read())
    
    print(f"📊 Tamaño total: {total_size / 1024 / 1024:.2f} MB")
    print(f"📊 Caracteres totales: {total_chars:,}")


def combine_corpus(output_file: str = "corpus_completo.txt"):
    """Combina todos los libros en un solo archivo."""
    output_path = os.path.join(DATA_DIR, output_file)
    
    print(f"\n📝 Combinando corpus en {output_file}...")
    
    all_text = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith('.txt') and filename != output_file:
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                # Limpiar headers de Gutenberg
                text = clean_gutenberg_text(text)
                all_text.append(text)
    
    combined = "\n\n".join(all_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    print(f"✅ Corpus combinado: {len(combined):,} caracteres")
    print(f"📁 Guardado en: {output_path}")
    
    return output_path


def clean_gutenberg_text(text: str) -> str:
    """Limpia los headers y footers de Project Gutenberg."""
    # Buscar inicio del contenido real
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]
    
    # Encontrar inicio
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Buscar el siguiente salto de línea después del marker
            newline_idx = text.find('\n', idx)
            if newline_idx != -1:
                start_idx = newline_idx + 1
            break
    
    # Encontrar fin
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
    
    return text[start_idx:end_idx].strip()


if __name__ == "__main__":
    download_all()
    combine_corpus()
