# Dockerfile para CPU
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copiar código fuente
COPY model.py .
COPY data_loader.py .
COPY server.py .
COPY don_quijote.txt .

# Copiar modelo entrenado
COPY gpt_don_quijote_saved_model/ ./gpt_don_quijote_saved_model/
COPY training_checkpoints/ ./training_checkpoints/

# Copiar frontend
COPY static/ ./static/

# Puerto
EXPOSE 8000

# Variables de entorno
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Comando
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
