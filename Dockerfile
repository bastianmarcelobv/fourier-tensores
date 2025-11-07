# Dockerfile para Aplicación de Procesamiento de Imágenes con Fourier y Tensores
# Incluye servidor SSH en puerto 822 y aplicación Dash en puerto 8050

FROM python:3.11-slim

# Metadatos
LABEL maintainer="Proyecto Fourier Tensores"
LABEL description="Aplicación web de procesamiento de imágenes con Transformada de Fourier y descomposiciones tensoriales"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    DASH_PORT=8052 \
    SSH_PORT=822

# Instalar dependencias del sistema incluyendo OpenSSH
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    supervisor \
    curl \
    vim \
    git \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configurar SSH
RUN mkdir -p /var/run/sshd && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# Crear usuario para SSH (opcional - puedes usar root o crear usuario específico)
RUN useradd -m -s /bin/bash fourier && \
    echo 'fourier:fourier2025' | chpasswd && \
    mkdir -p /home/fourier/.ssh && \
    chmod 700 /home/fourier/.ssh

# Configurar SSH para permitir login con password y en puerto 822
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#Port 22/Port 822/' /etc/ssh/sshd_config && \
    echo "AllowUsers fourier root" >> /etc/ssh/sshd_config

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivo de requisitos e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear estructura de directorios del proyecto
RUN mkdir -p /app/modules \
             /app/assets \
             /app/uploads \
             /app/outputs \
             /app/logs

# Copiar código de la aplicación
COPY app.py /app/
COPY modules/ /app/modules/
COPY assets/ /app/assets/

# Configurar permisos
RUN chmod -R 755 /app && \
    chown -R fourier:fourier /app

# Copiar script de inicio y supervisor config
COPY docker-entrypoint.sh /usr/local/bin/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Exponer puertos
# 822: SSH
# 8052: Aplicación Dash
EXPOSE 822 8052

# Healthcheck para verificar que la aplicación está funcionando
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8052/ || exit 1



# Punto de entrada
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
