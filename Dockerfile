FROM python:3.10-slim

# Evita prompts interactivos en apt
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponer puerto requerido por Render
EXPOSE 10000

# Ejecutar Streamlit
CMD ["streamlit", "run", "SkillTalk.py", "--server.port=10000", "--server.address=0.0.0.0"]

