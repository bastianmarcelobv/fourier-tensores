"""
Paquete de módulos para procesamiento de imágenes
con Transformada de Fourier y Descomposiciones Tensoriales
"""

from . import fourier_ops
from . import tensor_ops
from . import metrics
from . import visualizations

__all__ = ['fourier_ops', 'tensor_ops', 'metrics', 'visualizations']
__version__ = '1.0.0'
