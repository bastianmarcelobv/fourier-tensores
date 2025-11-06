"""
modules/fourier_ops.py
Módulo de Operaciones de Fourier

Este módulo implementa todas las operaciones relacionadas con la
Transformada de Fourier para procesamiento de imágenes sin usar
descomposiciones tensoriales (enfoque convencional).
"""

import numpy as np
from scipy import fft
import time


def apply_fft_2d(image_array):
    """
    Aplica FFT-2D a una imagen RGB canal por canal.
    
    Parámetros:
    -----------
    image_array : numpy.ndarray
        Imagen de forma (M, N, 3) donde M y N son las dimensiones
        espaciales y 3 representa los canales RGB.
    
    Retorna:
    --------
    tuple
        (fft_result, execution_time, operation_count)
        - fft_result: Array complejo de forma (M, N, 3)
        - execution_time: Tiempo en segundos
        - operation_count: Número estimado de operaciones
    """
    start_time = time.time()
    
    # Obtener dimensiones
    M, N, channels = image_array.shape
    
    # Aplicar FFT-2D a cada canal
    fft_channels = []
    for c in range(channels):
        # fft.fft2 computa la FFT bidimensional
        # fft.fftshift centra el espectro (componente DC al centro)
        fft_channel = fft.fftshift(fft.fft2(image_array[:, :, c]))
        fft_channels.append(fft_channel)
    
    fft_result = np.stack(fft_channels, axis=2)
    
    execution_time = time.time() - start_time
    
    # Estimar operaciones: O(MN log(MN)) por canal
    operation_count = int(channels * M * N * np.log2(M * N))
    
    return fft_result, execution_time, operation_count


def apply_lowpass_filter(fft_array, cutoff_frequency=0.1):
    """
    Aplica un filtro pasa-bajos Gaussiano en el dominio de Fourier.
    
    Este filtro atenúa las altas frecuencias (detalles y bordes)
    mientras preserva las bajas frecuencias (estructura general),
    produciendo un efecto de suavizado.
    
    Parámetros:
    -----------
    fft_array : numpy.ndarray
        Coeficientes de Fourier de forma (M, N, 3)
    cutoff_frequency : float
        Frecuencia de corte normalizada (0-1). Valores más bajos
        producen mayor suavizado.
    
    Retorna:
    --------
    numpy.ndarray
        Array filtrado de forma (M, N, 3)
    """
    M, N, channels = fft_array.shape
    
    # Crear malla de frecuencias centrada
    u = np.arange(-M // 2, M // 2)
    v = np.arange(-N // 2, N // 2)
    U, V = np.meshgrid(v, u)
    
    # Distancia desde el centro (frecuencia cero)
    D = np.sqrt(U**2 + V**2)
    
    # Filtro Gaussiano: exp(-D²/(2σ²))
    sigma = cutoff_frequency * max(M, N) / 2
    H = np.exp(-(D**2) / (2 * sigma**2))
    
    # Aplicar filtro a cada canal
    filtered = np.zeros_like(fft_array)
    for c in range(channels):
        filtered[:, :, c] = fft_array[:, :, c] * H
    
    return filtered


def apply_ifft_2d(fft_array):
    """
    Aplica la transformada inversa de Fourier para reconstruir la imagen.
    
    Parámetros:
    -----------
    fft_array : numpy.ndarray
        Coeficientes de Fourier filtrados de forma (M, N, 3)
    
    Retorna:
    --------
    tuple
        (reconstructed_image, execution_time)
        - reconstructed_image: Imagen reconstruida (M, N, 3)
        - execution_time: Tiempo de ejecución en segundos
    """
    start_time = time.time()
    
    M, N, channels = fft_array.shape
    
    # Aplicar IFFT-2D a cada canal
    reconstructed_channels = []
    for c in range(channels):
        # Primero, desplazar el espectro de vuelta
        ifft_channel = fft.ifft2(fft.ifftshift(fft_array[:, :, c]))
        # Tomar la parte real (eliminar pequeños residuos imaginarios)
        reconstructed_channels.append(np.real(ifft_channel))
    
    reconstructed = np.stack(reconstructed_channels, axis=2)
    
    execution_time = time.time() - start_time
    
    # Asegurar que los valores estén en el rango [0, 255]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed, execution_time


def calculate_memory_usage(image_array):
    """
    Calcula el uso de memoria aproximado del procesamiento sin tensores.
    
    Parámetros:
    -----------
    image_array : numpy.ndarray
        Imagen original
    
    Retorna:
    --------
    float
        Memoria estimada en MB
    """
    M, N, channels = image_array.shape
    
    # Memoria para la imagen original (uint8)
    original_mem = M * N * channels * 1  # 1 byte por píxel
    
    # Memoria para los coeficientes FFT (complex128)
    fft_mem = M * N * channels * 16  # 16 bytes por número complejo
    
    # Memoria total en MB
    total_mem_mb = (original_mem + fft_mem) / (1024 ** 2)
    
    return total_mem_mb


def process_image_standard(image_array, cutoff_frequency=0.15):
    """
    Función principal que procesa una imagen usando el método estándar
    (sin descomposiciones tensoriales).
    
    Esta función integra todos los pasos: FFT, filtrado, IFFT y métricas.
    
    Parámetros:
    -----------
    image_array : numpy.ndarray
        Imagen de entrada (M, N, 3)
    cutoff_frequency : float
        Frecuencia de corte para el filtro pasa-bajos
    
    Retorna:
    --------
    dict
        Diccionario con resultados y métricas:
        - 'image': Imagen procesada
        - 'fft_result': Coeficientes de Fourier
        - 'metrics': Diccionario con métricas de rendimiento
    """
    total_start = time.time()
    
    # Paso 1: Aplicar FFT-2D
    fft_result, fft_time, fft_ops = apply_fft_2d(image_array)
    
    # Paso 2: Aplicar filtro pasa-bajos
    filter_start = time.time()
    filtered_fft = apply_lowpass_filter(fft_result, cutoff_frequency)
    filter_time = time.time() - filter_start
    
    # Paso 3: Aplicar IFFT-2D
    processed_image, ifft_time = apply_ifft_2d(filtered_fft)
    
    # Calcular métricas totales
    total_time = time.time() - total_start
    memory_mb = calculate_memory_usage(image_array)
    
    metrics = {
        'method': 'Sin Tensores (Convencional)',
        'total_time': total_time,
        'fft_time': fft_time,
        'filter_time': filter_time,
        'ifft_time': ifft_time,
        'operations': fft_ops,
        'memory_mb': memory_mb,
        'complexity': f"O(MN log(MN))"
    }
    
    return {
        'image': processed_image,
        'fft_result': fft_result,
        'metrics': metrics
    }


def compute_psnr(original, processed):
    """
    Calcula el Peak Signal-to-Noise Ratio (PSNR) entre dos imágenes.
    
    El PSNR mide la calidad de reconstrucción. Valores más altos indican
    mejor calidad (menor pérdida de información).
    
    Parámetros:
    -----------
    original : numpy.ndarray
        Imagen original
    processed : numpy.ndarray
        Imagen procesada
    
    Retorna:
    --------
    float
        PSNR en dB
    """
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr
