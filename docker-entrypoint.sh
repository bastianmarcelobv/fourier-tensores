#!/bin/bash
# docker-entrypoint.sh - Script de inicio del contenedor
# Inicia el servidor SSH y la aplicación Dash usando supervisor

set -e

echo "======================================"
echo "Iniciando contenedor Fourier-Tensores"
echo "======================================"

# Generar claves SSH si no existen
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
    echo "Generando claves SSH..."
    ssh-keygen -A
fi

# Verificar estructura de directorios
echo "Verificando estructura de directorios..."
mkdir -p /app/uploads /app/outputs /app/logs
chown -R fourier:fourier /app/uploads /app/outputs /app/logs

# Mostrar información de configuración
echo ""
echo "Configuración del contenedor:"
echo "  - Usuario SSH: fourier / root"
echo "  - Password: fourier2025"
echo "  - Puerto SSH: 822"
echo "  - Puerto Dash: 8052"
echo ""
echo "Conexión SSH:"
echo "  ssh -p 822 fourier@localhost"
echo ""
echo "Aplicación Web:"
echo "  http://localhost:8052"
echo ""

# Iniciar supervisor que manejará SSH y Dash
echo "Iniciando servicios con Supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
