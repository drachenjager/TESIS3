FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

# Paso 1: instalar PyTorch CPU-only (~200 MB, sin CUDA)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       "torch==2.2.2+cpu" \
       --index-url https://download.pytorch.org/whl/cpu

# Paso 2: instalar el resto de dependencias
# (torch no está en requirements.txt, así que no hay conflicto)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# 1 worker · timeout 300 s (cubre entrenamiento de 7 modelos en 10 años de datos)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "wsgi:app"]
