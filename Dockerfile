# Dockerfile - Streamlit + MediaPipe (Python 3.12 slim)
# Build: docker build -t skilltalk-mediapipe:latest .

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# ---------- Apt dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libsndfile1 \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar solo los ficheros de requirements primero para aprovechar cache de docker
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el resto del repositorio
COPY . /app

# Dar permisos a start.sh
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Puerto de Streamlit
ENV PORT=8501

ENTRYPOINT ["/app/start.sh"]
