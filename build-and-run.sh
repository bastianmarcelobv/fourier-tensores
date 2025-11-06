#!/bin/bash
# build-and-run.sh - Script para construir y ejecutar el contenedor Docker

set -e

echo "=========================================="
echo "  Fourier-Tensores Docker Build & Run"
echo "=========================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Nombre del contenedor e imagen
IMAGE_NAME="fourier-tensores"
CONTAINER_NAME="fourier-app"

# Función para limpiar contenedores previos
cleanup() {
    echo -e "${YELLOW}Limpiando contenedores previos...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

# Función para construir la imagen
build_image() {
    echo -e "${BLUE}Construyendo imagen Docker...${NC}"
    docker build -t $IMAGE_NAME:latest .
    echo -e "${GREEN}✓ Imagen construida exitosamente${NC}"
}

# Función para ejecutar el contenedor
run_container() {
    echo -e "${BLUE}Iniciando contenedor...${NC}"
    docker run -d \
        --name $CONTAINER_NAME \
        -p 8052:8052 \
        -p 822:822 \
        -v fourier_uploads:/app/uploads \
        -v fourier_outputs:/app/outputs \
        -v fourier_logs:/app/logs \
        --restart unless-stopped \
        $IMAGE_NAME:latest
    
    echo -e "${GREEN}✓ Contenedor iniciado exitosamente${NC}"
}

# Función para mostrar información de acceso
show_info() {
    echo ""
    echo "=========================================="
    echo "  Información de Acceso"
    echo "=========================================="
    echo ""
    echo -e "${GREEN}Aplicación Web:${NC}"
    echo "  URL: http://localhost:8052"
    echo ""
    echo -e "${GREEN}Acceso SSH:${NC}"
    echo "  Comando: ssh -p 822 fourier@localhost"
    echo "  Password: fourier2025"
    echo ""
    echo -e "${GREEN}Comandos útiles:${NC}"
    echo "  Ver logs:        docker logs -f $CONTAINER_NAME"
    echo "  Detener:         docker stop $CONTAINER_NAME"
    echo "  Reiniciar:       docker restart $CONTAINER_NAME"
    echo "  Eliminar:        docker rm -f $CONTAINER_NAME"
    echo "  Shell interno:   docker exec -it $CONTAINER_NAME bash"
    echo ""
    echo "=========================================="
}

# Función para verificar el estado
check_status() {
    echo ""
    echo -e "${BLUE}Verificando estado del contenedor...${NC}"
    sleep 3
    
    if docker ps | grep -q $CONTAINER_NAME; then
        echo -e "${GREEN}✓ Contenedor ejecutándose correctamente${NC}"
        
        # Verificar aplicación web
        if curl -s http://localhost:8052 > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Aplicación web respondiendo${NC}"
        else
            echo -e "${YELLOW}⚠ Aplicación web aún iniciando...${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Contenedor no encontrado${NC}"
    fi
}

# Menú principal
case "${1:-all}" in
    build)
        build_image
        ;;
    run)
        cleanup
        run_container
        show_info
        check_status
        ;;
    rebuild)
        cleanup
        build_image
        run_container
        show_info
        check_status
        ;;
    all|*)
        cleanup
        build_image
        run_container
        show_info
        check_status
        ;;
esac

echo ""
echo -e "${GREEN}Proceso completado!${NC}"
