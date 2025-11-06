# Procesamiento de Imágenes con Fourier y Tensores - Docker

## Descripción del Proyecto

Este proyecto implementa una aplicación web completa para el procesamiento de imágenes mediante la Transformada de Fourier, comparando dos enfoques distintos:

1. **Método Convencional**: Transformada de Fourier bidimensional aplicada directamente a cada canal de color
2. **Método Tensorial**: Descomposición CP (CANDECOMP/PARAFAC) combinada con transformaciones de Fourier sobre los factores

La aplicación está completamente containerizada en Docker con acceso SSH para administración remota y una interfaz web interactiva desarrollada con Python Dash.

## Características

- **Aplicación Web Interactiva**: Interfaz Dash con visualizaciones en tiempo real usando Plotly
- **Análisis Comparativo**: Métricas detalladas de tiempo, memoria y complejidad computacional
- **Acceso SSH**: Puerto 822 para administración remota segura
- **Procesamiento Avanzado**: Filtros Gaussianos en dominio de frecuencias
- **Descomposición Tensorial**: Implementación de CP con TensorLy
- **Visualizaciones Científicas**: Espectros de Fourier, perfiles radiales, mapas de diferencias

## Arquitectura del Contenedor

```
┌─────────────────────────────────────────┐
│         Contenedor Docker               │
│                                         │
│  ┌──────────────┐    ┌──────────────┐ │
│  │   SSH        │    │   Dash App   │ │
│  │   Puerto 822 │    │   Puerto 8052│ │
│  └──────────────┘    └──────────────┘ │
│           │                  │          │
│           └─── Supervisor ───┘          │
│                                         │
│  ┌──────────────────────────────────┐ │
│  │  Módulos Python                  │ │
│  │  - fourier_ops.py                │ │
│  │  - tensor_ops.py                 │ │
│  │  - metrics.py                    │ │
│  │  - visualizations.py             │ │
│  └──────────────────────────────────┘ │
│                                         │
│  ┌──────────────────────────────────┐ │
│  │  Volúmenes Persistentes          │ │
│  │  - /app/uploads                  │ │
│  │  - /app/outputs                  │ │
│  │  - /app/logs                     │ │
│  └──────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Requisitos del Sistema

- **Docker**: Versión 20.10 o superior
- **RAM**: Mínimo 4GB disponibles, recomendado 8GB
- **Espacio en disco**: 2GB para la imagen y datos
- **Puertos**: 822 y 8052 deben estar disponibles

## Instalación y Uso

### Opción 1: Script Automático (Recomendado)

```bash
# Dar permisos de ejecución
chmod +x build-and-run.sh

# Construir y ejecutar todo
./build-and-run.sh

# O por pasos
./build-and-run.sh build    # Solo construir imagen
./build-and-run.sh run      # Solo ejecutar contenedor
./build-and-run.sh rebuild  # Limpiar, construir y ejecutar
```

### Opción 2: Comandos Docker Manuales

```bash
# Construir la imagen
docker build -t fourier-tensores:latest .

# Ejecutar el contenedor
docker run -d \
  --name fourier-app \
  -p 8052:8052 \
  -p 822:822 \
  -v fourier_uploads:/app/uploads \
  -v fourier_outputs:/app/outputs \
  -v fourier_logs:/app/logs \
  --restart unless-stopped \
  fourier-tensores:latest
```

## Acceso a la Aplicación

### Aplicación Web
```
URL: http://localhost:8052
```
Abre tu navegador y accede a la interfaz web interactiva.

### SSH
```bash
ssh -p 822 fourier@localhost
# Password: fourier2025
```

También puedes usar el usuario root:
```bash
ssh -p 822 root@localhost
# Password: (configurar en el Dockerfile si es necesario)
```

## Uso de la Aplicación

1. **Cargar Imagen**: Arrastra y suelta o selecciona una imagen JPG/PNG
2. **Configurar Parámetros**:
   - **Frecuencia de Corte**: Controla el nivel de suavizado (0.01-0.5)
   - **Rango CP**: Número de componentes tensoriales (5-50)
3. **Procesar**: Click en "Procesar Imagen"
4. **Analizar Resultados**: Visualiza las imágenes procesadas y métricas comparativas

## Estructura del Proyecto

```
fourier_tensor_docker/
├── Dockerfile                  # Definición del contenedor
├── docker-entrypoint.sh        # Script de inicio
├── supervisord.conf            # Configuración de supervisor
├── requirements.txt            # Dependencias Python
├── build-and-run.sh           # Script de construcción automatizada
├── app.py                     # Aplicación Dash principal
├── modules/
│   ├── __init__.py
│   ├── fourier_ops.py         # Operaciones de Fourier
│   ├── tensor_ops.py          # Operaciones tensoriales
│   ├── metrics.py             # Análisis de métricas
│   └── visualizations.py      # Generación de gráficos
└── assets/
    └── style.css              # Estilos personalizados
```

## Comandos Útiles

### Gestión del Contenedor
```bash
# Ver logs en tiempo real
docker logs -f fourier-app

# Detener contenedor
docker stop fourier-app

# Reiniciar contenedor
docker restart fourier-app

# Eliminar contenedor
docker rm -f fourier-app

# Acceder a shell interno
docker exec -it fourier-app bash
```

### Gestión de Volúmenes
```bash
# Listar volúmenes
docker volume ls

# Inspeccionar volumen
docker volume inspect fourier_uploads

# Limpiar volúmenes no usados
docker volume prune
```

### Monitoreo
```bash
# Estadísticas de recursos
docker stats fourier-app

# Verificar salud
docker inspect --format='{{.State.Health.Status}}' fourier-app
```

## Puertos Expuestos

| Puerto | Servicio | Descripción |
|--------|----------|-------------|
| 822    | SSH      | Acceso remoto para administración |
| 8052   | Dash App | Aplicación web interactiva |

## Volúmenes

| Volumen | Ruta Interna | Propósito |
|---------|--------------|-----------|
| fourier_uploads | /app/uploads | Imágenes cargadas por usuarios |
| fourier_outputs | /app/outputs | Imágenes procesadas |
| fourier_logs | /app/logs | Logs de la aplicación |

## Complejidad Computacional

### Método Sin Tensores
- **Temporal**: O(MN log(MN))
- **Espacial**: O(MN)
- **Mejor para**: Imágenes pequeñas, procesamiento único

### Método Con Tensores
- **Temporal**: O(I × MNR) donde I ≈ 50 iteraciones, R = rango
- **Espacial**: O(R(M+N))
- **Mejor para**: Imágenes grandes, múltiples operaciones

## Solución de Problemas

### El contenedor no inicia
```bash
# Verificar logs
docker logs fourier-app

# Verificar puertos ocupados
lsof -i :8052
lsof -i :822
```

### No se puede conectar por SSH
```bash
# Verificar que el servicio SSH está corriendo
docker exec fourier-app supervisorctl status sshd

# Reiniciar SSH
docker exec fourier-app supervisorctl restart sshd
```

### Error de memoria
```bash
# Aumentar límites de Docker
docker run --memory="8g" --memory-swap="8g" ...
```

## Seguridad

### Cambiar Password SSH
1. Acceder al contenedor:
   ```bash
   docker exec -it fourier-app bash
   ```

2. Cambiar password:
   ```bash
   passwd fourier
   ```

### Deshabilitar Root SSH
Editar `/etc/ssh/sshd_config` dentro del contenedor y cambiar:
```
PermitRootLogin no
```

## Desarrollo

### Ejecutar en Modo Desarrollo
```bash
docker run -it \
  -p 8052:8052 \
  -p 822:822 \
  -v $(pwd):/app \
  fourier-tensores:latest \
  bash
```

### Instalar Dependencias Adicionales
```bash
docker exec -it fourier-app bash
pip install nueva-libreria --break-system-packages
```

## Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Realiza tus cambios
4. Envía un pull request

## Licencia

Este proyecto está basado en el documento técnico de investigación sobre procesamiento de imágenes con Fourier y tensores.

## Referencias

- TensorLy: http://tensorly.org
- Dash by Plotly: https://dash.plotly.com/
- NumPy FFT: https://numpy.org/doc/stable/reference/routines.fft.html

## Soporte

Para reportar problemas o solicitar features, por favor abre un issue en el repositorio del proyecto.

---

**Versión**: 1.0.0  
**Última actualización**: Octubre 2025
