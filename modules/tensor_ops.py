"""
modules/tensor_ops.py
Módulo de Operaciones Tensoriales

Este módulo implementa el procesamiento de imágenes mediante
descomposiciones tensoriales CP (CANDECOMP/PARAFAC) combinadas
con la Transformada de Fourier.
"""

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from scipy import fft
import time


def decompose_image_tensor(image_array, rank=10):
    """
    Descompone una imagen usando CP decomposition.
    
    La descomposición CP factoriza el tensor imagen como:
    X ≈ Σ[r=1 to R] a_r ⊗ b_r ⊗ c_r
    
    donde cada componente representa un patrón espacial y de color.
    
    Parámetros:
    -----------
    image_array : numpy.ndarray
        Imagen de entrada (M, N, 3)
    rank : int
        Rango de la descomposición (número de componentes)
    
    Retorna:
    --------
    tuple
        (tensor_factors, execution_time, operations)
        - tensor_factors: Factores de la descomposición CP
        - execution_time: Tiempo de ejecución
        - operations: Estimación de operaciones realizadas
    """
    start_time = time.time()
    
    # Normalizar imagen a [0, 1] para mejor estabilidad numérica
    img_normalized = image_array.astype(np.float64) / 255.0
    
    # Realizar descomposición CP
    # parafac devuelve (weights, factors) donde factors es una lista de matrices
    try:
        tensor_factors = parafac(img_normalized, rank=rank, init='random', 
                                n_iter_max=50, tol=1e-6)
    except Exception as e:
        print(f"Error en descomposición CP: {e}")
        # Intentar con parámetros más conservadores
        tensor_factors = parafac(img_normalized, rank=rank, init='svd', 
                                n_iter_max=30, tol=1e-5)
    
    exec_time = time.time() - start_time
    
    # Estimar operaciones: O(I × M × N × R)
    M, N, _ = image_array.shape
    iterations = 50  # Estimación de iteraciones CP
    ops = iterations * M * N * rank
    
    return tensor_factors, exec_time, ops


def apply_fourier_filter_to_reconstructed(reconstructed_normalized, cutoff_frequency=0.15):
    """
    Aplica filtro de Fourier a la imagen reconstruida desde los factores CP.
    
    Este es el enfoque correcto: primero reconstruimos la imagen completa desde
    los factores CP, luego aplicamos FFT-2D y filtrado como en el método convencional.
    
    Parámetros:
    -----------
    reconstructed_normalized : numpy.ndarray
        Imagen reconstruida normalizada [0, 1] de forma (M, N, 3)
    cutoff_frequency : float
        Frecuencia de corte para el filtro pasa-bajos
    
    Retorna:
    --------
    numpy.ndarray
        Imagen filtrada normalizada [0, 1]
    """
    M, N, channels = reconstructed_normalized.shape
    
    # Crear filtro Gaussiano 2D
    u = np.arange(-M // 2, M // 2)
    v = np.arange(-N // 2, N // 2)
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    
    sigma = cutoff_frequency * max(M, N) / 2
    H = np.exp(-(D**2) / (2 * sigma**2))
    
    # Aplicar FFT-2D, filtrar, e IFFT-2D a cada canal
    filtered_channels = []
    for c in range(channels):
        # FFT-2D con shift para centrar el espectro
        fft_channel = fft.fftshift(fft.fft2(reconstructed_normalized[:, :, c]))
        
        # Aplicar filtro
        fft_filtered = fft_channel * H
        
        # IFFT-2D para volver al dominio espacial
        ifft_channel = fft.ifft2(fft.ifftshift(fft_filtered))
        
        # Tomar parte real y asegurar que esté en [0, 1]
        filtered_channel = np.real(ifft_channel)
        filtered_channel = np.clip(filtered_channel, 0, 1)
        
        filtered_channels.append(filtered_channel)
    
    # Recombinar canales
    filtered_image = np.stack(filtered_channels, axis=2)
    
    return filtered_image


def reconstruct_from_cp(tensor_factors):
    """
    Reconstruye el tensor (imagen) a partir de los factores CP.
    
    Utiliza la fórmula: X = Σ[r=1 to R] weights[r] * (a_r ⊗ b_r ⊗ c_r)
    
    Parámetros:
    -----------
    tensor_factors : tuple
        (weights, factors) de la descomposición CP
    
    Retorna:
    --------
    numpy.ndarray
        Imagen reconstruida normalizada [0, 1] de forma (M, N, 3)
    """
    # Reconstruir usando TensorLy - esto devuelve valores normalizados [0, 1]
    reconstructed = tl.cp_to_tensor(tensor_factors)
    
    # Asegurar que los valores estén en el rango [0, 1]
    reconstructed = np.clip(reconstructed, 0, 1)
    
    return reconstructed


def convert_to_uint8(image_normalized):
    """
    Convierte imagen normalizada [0, 1] a formato uint8 [0, 255].
    
    Parámetros:
    -----------
    image_normalized : numpy.ndarray
        Imagen con valores en [0, 1]
    
    Retorna:
    --------
    numpy.ndarray
        Imagen en formato uint8 [0, 255]
    """
    image_uint8 = (image_normalized * 255.0).astype(np.uint8)
    return image_uint8


def calculate_memory_tensor(image_array, rank):
    """
    Calcula el uso de memoria de la descomposición tensorial.
    
    Parámetros:
    -----------
    image_array : numpy.ndarray
        Imagen original
    rank : int
        Rango de la descomposición
    
    Retorna:
    --------
    float
        Memoria estimada en MB
    """
    M, N, channels = image_array.shape
    
    # Memoria para los factores CP
    # Factor 1: M x R, Factor 2: N x R, Factor 3: 3 x R
    # Cada elemento es float64 (8 bytes)
    factor_mem = (M * rank + N * rank + channels * rank) * 8
    
    # Memoria adicional para pesos (R elementos)
    weights_mem = rank * 8
    
    # Memoria total en MB
    total_mem_mb = (factor_mem + weights_mem) / (1024 ** 2)
    
    return total_mem_mb


def process_image_tensor(image_array, cutoff_frequency=0.15, rank=20):
    """
    Función principal que procesa una imagen usando descomposición tensorial CP.
    
    ENFOQUE CORREGIDO:
    1. Descompone la imagen en factores CP (compresión)
    2. Reconstruye la imagen completa desde los factores
    3. Aplica FFT-2D y filtrado en dominio de frecuencias
    4. Aplica IFFT-2D para obtener resultado final
    
    La ventaja del método tensorial es la compresión en el paso 1,
    que usa mucha menos memoria que almacenar la imagen completa.
    
    Parámetros:
    -----------
    image_array : numpy.ndarray
        Imagen de entrada (M, N, 3) en formato uint8 [0, 255]
    cutoff_frequency : float
        Frecuencia de corte para el filtro pasa-bajos
    rank : int
        Rango de la descomposición tensorial
    
    Retorna:
    --------
    dict
        Diccionario con resultados y métricas
    """
    total_start = time.time()
    
    # Paso 1: Descomposición CP
    print(f"  - Descomponiendo tensor con rank={rank}...")
    tensor_factors, decomp_time, decomp_ops = decompose_image_tensor(image_array, rank)
    
    # Paso 2: Reconstruir imagen desde factores CP
    print(f"  - Reconstruyendo imagen desde factores...")
    recon_start = time.time()
    reconstructed_normalized = reconstruct_from_cp(tensor_factors)
    recon_time = time.time() - recon_start
    
    # Paso 3: Aplicar filtro de Fourier a la imagen reconstruida
    print(f"  - Aplicando filtro de Fourier 2D...")
    filter_start = time.time()
    filtered_normalized = apply_fourier_filter_to_reconstructed(
        reconstructed_normalized, 
        cutoff_frequency
    )
    filter_time = time.time() - filter_start
    
    # Paso 4: Convertir a formato uint8 para visualización
    print(f"  - Convirtiendo a formato de imagen...")
    convert_start = time.time()
    processed_image = convert_to_uint8(filtered_normalized)
    convert_time = time.time() - convert_start
    
    # Calcular métricas
    total_time = time.time() - total_start
    memory_mb = calculate_memory_tensor(image_array, rank)
    
    metrics = {
        'method': 'Con Tensores (CP)',
        'total_time': total_time,
        'decomposition_time': decomp_time,
        'reconstruction_time': recon_time,
        'filter_time': filter_time,
        'conversion_time': convert_time,
        'operations': decomp_ops,
        'memory_mb': memory_mb,
        'rank': rank,
        'complexity': f"O(I × MNR), R={rank}"
    }
    
    return {
        'image': processed_image,
        'factors': tensor_factors,
        'reconstructed': reconstructed_normalized,
        'metrics': metrics
    }


def compute_compression_ratio(image_array, rank):
    """
    Calcula la razón de compresión de la descomposición tensorial.
    
    Parámetros:
    -----------
    image_array : numpy.ndarray
        Imagen original
    rank : int
        Rango de descomposición
    
    Retorna:
    --------
    float
        Razón de compresión
    """
    M, N, channels = image_array.shape
    
    original_size = M * N * channels
    compressed_size = (M + N + channels) * rank
    
    compression_ratio = original_size / compressed_size
    
    return compression_ratio
