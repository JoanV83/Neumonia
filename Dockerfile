# Imagen base ligera para CPU
FROM python:3.11-slim

# Configurar entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema necesarias para OpenCV-headless y TensorFlow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements.txt y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente completo
COPY src/ ./src/
COPY scripts/ ./scripts/

# Crear directorios para modelos, datos y reportes
RUN mkdir -p models data reports

# Configurar entrada usando el módulo correcto
ENTRYPOINT ["python", "-m", "src.visualizations.integrator"]
CMD ["--help"]
