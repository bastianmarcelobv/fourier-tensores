"""
modules/metrics.py
Módulo de Métricas y Análisis Comparativo

Este módulo proporciona funciones para calcular, comparar y
presentar métricas de rendimiento de ambos métodos de procesamiento.
"""

import numpy as np
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd


def compare_methods(metrics_standard, metrics_tensor):
    """
    Genera un análisis comparativo detallado entre ambos métodos.
    
    Parámetros:
    -----------
    metrics_standard : dict
        Métricas del método estándar
    metrics_tensor : dict
        Métricas del método tensorial
    
    Retorna:
    --------
    dict
        Diccionario con comparaciones:
        - speedup: Factor de mejora en velocidad (puede ser <1 si es más lento)
        - memory_reduction: Porcentaje de reducción de memoria
        - ops_ratio: Ratio de operaciones
    """
    # Evitar división por cero
    if metrics_tensor['total_time'] == 0:
        speedup = float('inf')
    else:
        speedup = metrics_standard['total_time'] / metrics_tensor['total_time']
    
    memory_reduction = 100 * (1 - metrics_tensor['memory_mb'] / metrics_standard['memory_mb'])
    ops_ratio = metrics_tensor['operations'] / metrics_standard['operations']
    
    comparison = {
        'speedup': speedup,
        'memory_reduction': memory_reduction,
        'ops_ratio': ops_ratio,
        'time_diff': metrics_tensor['total_time'] - metrics_standard['total_time'],
        'memory_diff': metrics_tensor['memory_mb'] - metrics_standard['memory_mb']
    }
    
    return comparison


def create_metrics_table(metrics_standard, metrics_tensor):
    """
    Crea una tabla HTML con Bootstrap para mostrar métricas comparativas.
    
    Parámetros:
    -----------
    metrics_standard : dict
        Métricas del método estándar
    metrics_tensor : dict
        Métricas del método tensorial
    
    Retorna:
    --------
    dbc.Table
        Componente de tabla Dash Bootstrap
    """
    # Calcular comparaciones
    comparison = compare_methods(metrics_standard, metrics_tensor)
    
    # Crear datos para la tabla
    data = [
        {
            'Métrica': 'Tiempo Total',
            'Sin Tensores': f"{metrics_standard['total_time']:.4f} s",
            'Con Tensores': f"{metrics_tensor['total_time']:.4f} s",
            'Diferencia': f"{comparison['time_diff']:+.4f} s"
        },
        {
            'Métrica': 'Uso de Memoria',
            'Sin Tensores': f"{metrics_standard['memory_mb']:.2f} MB",
            'Con Tensores': f"{metrics_tensor['memory_mb']:.2f} MB",
            'Diferencia': f"{comparison['memory_reduction']:+.1f}%"
        },
        {
            'Métrica': 'Operaciones (est.)',
            'Sin Tensores': f"{metrics_standard['operations']:,}",
            'Con Tensores': f"{metrics_tensor['operations']:,}",
            'Diferencia': f"{comparison['ops_ratio']:.2f}x"
        },
        {
            'Métrica': 'Complejidad',
            'Sin Tensores': metrics_standard['complexity'],
            'Con Tensores': metrics_tensor['complexity'],
            'Diferencia': '-'
        }
    ]
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Crear tabla Bootstrap
    table = dbc.Table.from_dataframe(
        df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className='table-sm'
    )
    
    # Agregar análisis textual
    analysis = create_analysis_text(comparison, metrics_standard, metrics_tensor)
    
    return html.Div([
        table,
        html.Hr(),
        analysis
    ])


def create_analysis_text(comparison, metrics_standard, metrics_tensor):
    """
    Genera un análisis textual de los resultados.
    
    Parámetros:
    -----------
    comparison : dict
        Diccionario con comparaciones
    metrics_standard : dict
        Métricas del método estándar
    metrics_tensor : dict
        Métricas del método tensorial
    
    Retorna:
    --------
    dbc.Alert
        Componente con el análisis textual
    """
    # Determinar cuál método es más rápido
    if comparison['speedup'] > 1:
        speed_text = f"El método estándar es {comparison['speedup']:.2f}x más rápido."
        speed_color = "info"
    else:
        speed_text = f"El método tensorial es {1/comparison['speedup']:.2f}x más rápido."
        speed_color = "success"
    
    # Análisis de memoria
    if comparison['memory_reduction'] > 0:
        memory_text = f"El método tensorial reduce el uso de memoria en {comparison['memory_reduction']:.1f}%."
        memory_icon = "✓"
    else:
        memory_text = f"El método tensorial usa {-comparison['memory_reduction']:.1f}% más memoria."
        memory_icon = "⚠"
    
    # Recomendación
    if comparison['memory_reduction'] > 50:
        recommendation = ("El método tensorial es altamente recomendado para imágenes grandes "
                         "o procesamiento batch donde la memoria es limitada.")
        rec_color = "success"
    elif comparison['speedup'] > 2:
        recommendation = ("El método estándar es preferible para procesamiento en tiempo real "
                         "de imágenes individuales donde la velocidad es crítica.")
        rec_color = "info"
    else:
        recommendation = ("Ambos métodos tienen rendimiento comparable. La elección depende "
                         "de los requisitos específicos de la aplicación.")
        rec_color = "warning"
    
    analysis_content = html.Div([
        html.H6("Análisis Comparativo:", className="mb-3"),
        html.P([
            html.Strong("Velocidad: "),
            speed_text
        ]),
        html.P([
            html.Strong("Memoria: "),
            f"{memory_icon} ",
            memory_text
        ]),
        html.P([
            html.Strong("Operaciones: "),
            f"El método tensorial realiza {comparison['ops_ratio']:.2f}x operaciones respecto al estándar."
        ]),
        html.Hr(),
        html.P([
            html.Strong("Recomendación: "),
            recommendation
        ])
    ])
    
    return dbc.Alert(analysis_content, color=rec_color, className="mt-3")


def calculate_quality_metrics(original, processed):
    """
    Calcula métricas de calidad de imagen.
    
    Parámetros:
    -----------
    original : numpy.ndarray
        Imagen original
    processed : numpy.ndarray
        Imagen procesada
    
    Retorna:
    --------
    dict
        Diccionario con métricas de calidad:
        - psnr: Peak Signal-to-Noise Ratio en dB
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
    """
    # Convertir a float para cálculos precisos
    orig = original.astype(float)
    proc = processed.astype(float)
    
    # MSE
    mse = np.mean((orig - proc) ** 2)
    
    # MAE
    mae = np.mean(np.abs(orig - proc))
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {
        'psnr': psnr,
        'mse': mse,
        'mae': mae
    }


def format_time(seconds):
    """
    Formatea tiempo en segundos a un string legible.
    
    Parámetros:
    -----------
    seconds : float
        Tiempo en segundos
    
    Retorna:
    --------
    str
        Tiempo formateado
    """
    if seconds < 0.001:
        return f"{seconds*1000000:.1f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.3f} s"


def format_memory(mb):
    """
    Formatea memoria en MB a un string legible.
    
    Parámetros:
    -----------
    mb : float
        Memoria en megabytes
    
    Retorna:
    --------
    str
        Memoria formateada
    """
    if mb < 1:
        return f"{mb*1024:.2f} KB"
    elif mb < 1024:
        return f"{mb:.2f} MB"
    else:
        return f"{mb/1024:.2f} GB"
