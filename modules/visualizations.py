"""
modules/visualizations.py
Módulo de Visualizaciones

Este módulo proporciona funciones para crear visualizaciones
interactivas de los resultados del procesamiento de imágenes.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_comparison_figure(original, processed_standard, processed_tensor):
    """
    Crea una figura comparativa mostrando las tres imágenes lado a lado.
    
    Parámetros:
    -----------
    original : numpy.ndarray
        Imagen original
    processed_standard : numpy.ndarray
        Imagen procesada con método estándar
    processed_tensor : numpy.ndarray
        Imagen procesada con método tensorial
    
    Retorna:
    --------
    plotly.graph_objects.Figure
        Figura con las tres imágenes
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Original', 'Sin Tensores', 'Con Tensores'),
        horizontal_spacing=0.05
    )
    
    # Imagen original
    fig.add_trace(
        go.Image(z=original),
        row=1, col=1
    )
    
    # Imagen procesada sin tensores
    fig.add_trace(
        go.Image(z=processed_standard),
        row=1, col=2
    )
    
    # Imagen procesada con tensores
    fig.add_trace(
        go.Image(z=processed_tensor),
        row=1, col=3
    )
    
    # Actualizar layout
    fig.update_layout(
        title_text="Comparación Visual de Métodos",
        title_x=0.5,
        title_font_size=18,
        showlegend=False,
        height=400,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    
    # Ocultar ejes
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    
    return fig


def create_spectrum_figure(fft_result):
    """
    Crea una visualización del espectro de magnitud de Fourier.
    
    Parámetros:
    -----------
    fft_result : numpy.ndarray
        Coeficientes de Fourier (M, N, 3)
    
    Retorna:
    --------
    plotly.graph_objects.Figure
        Figura con el espectro de magnitud
    """
    # Calcular magnitud del espectro (promedio de los 3 canales)
    magnitude = np.mean(np.abs(fft_result), axis=2)
    
    # Aplicar escala logarítmica para mejor visualización
    magnitude_log = np.log1p(magnitude)
    
    fig = go.Figure(data=go.Heatmap(
        z=magnitude_log,
        colorscale='Viridis',
        colorbar=dict(title="Log Magnitud")
    ))
    
    fig.update_layout(
        title="Espectro de Frecuencias de Fourier (Magnitud)",
        title_x=0.5,
        xaxis_title="Frecuencia Horizontal",
        yaxis_title="Frecuencia Vertical",
        height=500,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # Mantener aspecto cuadrado
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def create_difference_map(image1, image2):
    """
    Crea un mapa de diferencias entre dos imágenes.
    
    Parámetros:
    -----------
    image1 : numpy.ndarray
        Primera imagen
    image2 : numpy.ndarray
        Segunda imagen
    
    Retorna:
    --------
    plotly.graph_objects.Figure
        Figura con el mapa de diferencias
    """
    # Calcular diferencia absoluta
    diff = np.abs(image1.astype(float) - image2.astype(float))
    
    # Convertir a escala de grises (promedio de canales)
    diff_gray = np.mean(diff, axis=2)
    
    fig = go.Figure(data=go.Heatmap(
        z=diff_gray,
        colorscale='Hot',
        colorbar=dict(title="Diferencia")
    ))
    
    fig.update_layout(
        title="Mapa de Diferencias entre Métodos",
        title_x=0.5,
        xaxis_title="X",
        yaxis_title="Y",
        height=500
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def create_performance_bars(metrics_standard, metrics_tensor):
    """
    Crea gráfico de barras comparando rendimiento.
    
    Parámetros:
    -----------
    metrics_standard : dict
        Métricas del método estándar
    metrics_tensor : dict
        Métricas del método tensorial
    
    Retorna:
    --------
    plotly.graph_objects.Figure
        Figura con barras comparativas
    """
    categories = ['Tiempo (s)', 'Memoria (MB)', 'Operaciones (×10⁶)']
    
    standard_values = [
        metrics_standard['total_time'],
        metrics_standard['memory_mb'],
        metrics_standard['operations'] / 1e6
    ]
    
    tensor_values = [
        metrics_tensor['total_time'],
        metrics_tensor['memory_mb'],
        metrics_tensor['operations'] / 1e6
    ]
    
    fig = go.Figure(data=[
        go.Bar(name='Sin Tensores', x=categories, y=standard_values, marker_color='#3498db'),
        go.Bar(name='Con Tensores', x=categories, y=tensor_values, marker_color='#e74c3c')
    ])
    
    fig.update_layout(
        title="Comparación de Rendimiento",
        title_x=0.5,
        barmode='group',
        height=400,
        yaxis_title="Valor",
        legend=dict(x=0.7, y=1),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def create_frequency_profile(fft_result):
    """
    Crea un perfil de frecuencias mostrando la energía en función
    de la distancia radial desde el centro.
    
    Parámetros:
    -----------
    fft_result : numpy.ndarray
        Coeficientes de Fourier
    
    Retorna:
    --------
    plotly.graph_objects.Figure
        Figura con perfil radial de frecuencias
    """
    # Calcular magnitud promedio
    magnitude = np.mean(np.abs(fft_result), axis=2)
    
    M, N = magnitude.shape
    center_y, center_x = M // 2, N // 2
    
    # Crear malla de distancias radiales
    y, x = np.ogrid[:M, :N]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
    
    # Calcular perfil radial
    max_dist = int(np.sqrt(center_x**2 + center_y**2))
    radial_profile = np.zeros(max_dist)
    
    for d in range(max_dist):
        mask = distances == d
        if np.any(mask):
            radial_profile[d] = np.mean(magnitude[mask])
    
    fig = go.Figure(data=go.Scatter(
        x=np.arange(max_dist),
        y=np.log1p(radial_profile),
        mode='lines',
        line=dict(color='#2ecc71', width=2)
    ))
    
    fig.update_layout(
        title="Perfil Radial de Frecuencias",
        title_x=0.5,
        xaxis_title="Distancia desde Centro (píxeles)",
        yaxis_title="Log(Energía)",
        height=400,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def create_convergence_plot(iteration_errors):
    """
    Crea gráfico de convergencia para la descomposición CP.
    
    Parámetros:
    -----------
    iteration_errors : list
        Lista de errores por iteración
    
    Retorna:
    --------
    plotly.graph_objects.Figure
        Figura mostrando convergencia
    """
    fig = go.Figure(data=go.Scatter(
        y=iteration_errors,
        mode='lines+markers',
        line=dict(color='#9b59b6', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Convergencia de la Descomposición CP",
        title_x=0.5,
        xaxis_title="Iteración",
        yaxis_title="Error de Reconstrucción",
        height=400,
        yaxis_type="log",
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig
