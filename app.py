"""
app.py - Aplicación Principal Dash
Procesamiento de Imágenes con Transformada de Fourier y Descomposiciones Tensoriales

Esta aplicación permite comparar el procesamiento de imágenes usando:
1. Transformada de Fourier convencional (sin tensores)
2. Descomposición tensorial CP + Fourier

Basado en el documento técnico del proyecto.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import numpy as np
from PIL import Image

# Importar módulos propios
from modules import fourier_ops, tensor_ops, metrics, visualizations

# Inicializar aplicación Dash con tema Bootstrap
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Procesamiento de Imágenes - Fourier & Tensores"
)

server = app.server

# Layout de la aplicación
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1(
                "Procesamiento de Imágenes con Transformada de Fourier y Tensores",
                className="text-center my-4",
                style={'color': '#2c3e50'}
            ),
            html.H5(
                "Análisis Comparativo: Implementación Convencional vs Descomposición Tensorial CP",
                className="text-center text-muted mb-4"
            )
        ])
    ]),
    
    html.Hr(),
    
    # Sección de carga de imagen
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("1. Cargar Imagen")),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            html.I(className="bi bi-cloud-upload", style={'fontSize': '48px'}),
                            html.Br(),
                            'Arrastra y suelta o ',
                            html.A('selecciona una imagen')
                        ]),
                        style={
                            'width': '100%',
                            'height': '150px',
                            'lineHeight': '150px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'cursor': 'pointer'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status', className='mt-3')
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Sección de parámetros
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("2. Configurar Parámetros")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Frecuencia de Corte (0-1):"),
                            dcc.Slider(
                                id='cutoff-slider',
                                min=0.01,
                                max=0.5,
                                step=0.01,
                                value=0.15,
                                marks={0.01: '0.01', 0.15: '0.15', 0.3: '0.3', 0.5: '0.5'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=6),
                        dbc.Col([
                            html.Label("Rango CP (componentes tensoriales):"),
                            dcc.Slider(
                                id='rank-slider',
                                min=5,
                                max=50,
                                step=5,
                                value=20,
                                marks={5: '5', 20: '20', 35: '35', 50: '50'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=6)
                    ]),
                    html.Br(),
                    dbc.Button(
                        "Procesar Imagen",
                        id="process-button",
                        color="primary",
                        size="lg",
                        className="w-100",
                        disabled=True
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    # Sección de resultados
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("3. Resultados y Comparación")),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-results",
                        type="default",
                        children=html.Div(id='results-container')
                    )
                ])
            ])
        ], width=12)
    ]),
    
    # Almacenamiento de datos
    dcc.Store(id='image-data'),
    dcc.Store(id='processed-data')
    
], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '20px'})


# Callbacks
@callback(
    [Output('upload-status', 'children'),
     Output('image-data', 'data'),
     Output('process-button', 'disabled')],
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def upload_image(contents, filename):
    """Procesar imagen cargada y almacenarla"""
    if contents is None:
        return html.Div("No se ha cargado ninguna imagen.", className="text-muted"), None, True
    
    try:
        # Decodificar imagen
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(io.BytesIO(decoded))
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar si es muy grande (optimización)
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convertir a array numpy
        img_array = np.array(img)
        
        status = dbc.Alert([
            html.I(className="bi bi-check-circle-fill me-2"),
            f"Imagen cargada: {filename} - Dimensiones: {img_array.shape}"
        ], color="success")
        
        return status, {'array': img_array.tolist(), 'filename': filename}, False
        
    except Exception as e:
        error = dbc.Alert([
            html.I(className="bi bi-exclamation-triangle-fill me-2"),
            f"Error al cargar imagen: {str(e)}"
        ], color="danger")
        return error, None, True


@callback(
    Output('results-container', 'children'),
    Input('process-button', 'n_clicks'),
    State('image-data', 'data'),
    State('cutoff-slider', 'value'),
    State('rank-slider', 'value'),
    prevent_initial_call=True
)
def process_image(n_clicks, image_data, cutoff, rank):
    """Procesar imagen con ambos métodos y mostrar comparación"""
    if image_data is None:
        return html.Div("No hay imagen para procesar.")
    
    try:
        # Recuperar imagen
        img_array = np.array(image_data['array'])
        
        # Procesar con método convencional (sin tensores)
        print("Procesando con método convencional...")
        result_standard = fourier_ops.process_image_standard(img_array, cutoff)
        
        # Procesar con método tensorial
        print("Procesando con descomposición tensorial...")
        result_tensor = tensor_ops.process_image_tensor(img_array, cutoff, rank)
        
        # Generar visualizaciones
        fig_comparison = visualizations.create_comparison_figure(
            img_array,
            result_standard['image'],
            result_tensor['image']
        )
        
        # Crear tabla de métricas
        metrics_table = metrics.create_metrics_table(
            result_standard['metrics'],
            result_tensor['metrics']
        )
        
        # Crear gráfico de espectro de Fourier
        fig_spectrum = visualizations.create_spectrum_figure(
            result_standard['fft_result']
        )
        
        return html.Div([
            # Comparación de imágenes
            dbc.Row([
                dbc.Col([
                    html.H5("Comparación Visual", className="text-center mb-3"),
                    dcc.Graph(figure=fig_comparison, config={'displayModeBar': False})
                ], width=12)
            ], className="mb-4"),
            
            # Tabla de métricas
            dbc.Row([
                dbc.Col([
                    html.H5("Análisis de Rendimiento", className="text-center mb-3"),
                    metrics_table
                ], width=12)
            ], className="mb-4"),
            
            # Espectro de Fourier
            dbc.Row([
                dbc.Col([
                    html.H5("Espectro de Frecuencias", className="text-center mb-3"),
                    dcc.Graph(figure=fig_spectrum, config={'displayModeBar': False})
                ], width=12)
            ])
        ])
        
    except Exception as e:
        return dbc.Alert([
            html.H5("Error en el procesamiento", className="alert-heading"),
            html.P(f"Detalles: {str(e)}")
        ], color="danger")


if __name__ == '__main__':
    print("=" * 60)
    print("Iniciando aplicación Dash - Fourier & Tensores")
    print("=" * 60)
    print(f"Accede a la aplicación en: http://0.0.0.0:8052")
    print("=" * 60)
    
    # Ejecutar servidor
    app.run_server(
        host='0.0.0.0',
        port=8052,
        debug=True
    )
