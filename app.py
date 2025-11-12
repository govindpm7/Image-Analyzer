"""
Dash Frontend for Image Analyzer
"""
import sys
import os
# Add custom package location for torch (to avoid Windows long path issues)
sys.path.insert(0, r'C:\py_pkgs')

import base64
import io
from pathlib import Path

import cv2
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import numpy as np
from PIL import Image

from models.blur_detector import BlurDetector
from models.aesthetic_scorer import AestheticScorer
from models.enhancer import LowLightEnhancer
from utils.image_processing import ImageProcessor

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Image Analyzer"

# Initialize models (lazy loading)
blur_detector = None
aesthetic_scorer = None
enhancer = None
image_processor = ImageProcessor()

def get_models():
    """Lazy load models on first use"""
    global blur_detector, aesthetic_scorer, enhancer
    
    if blur_detector is None:
        blur_detector = BlurDetector()
        # Try to load weights if available
        weights_path = Path('weights/blur_detector_best.pth')
        if weights_path.exists():
            blur_detector.load_weights(str(weights_path))
    
    if aesthetic_scorer is None:
        aesthetic_scorer = AestheticScorer()
        weights_path = Path('weights/aesthetic_scorer_best.pth')
        if weights_path.exists():
            aesthetic_scorer.load_weights(str(weights_path))
    
    if enhancer is None:
        enhancer = LowLightEnhancer()
        weights_path = Path('weights/enhancer_best.pth')
        if weights_path.exists():
            enhancer.load_weights(str(weights_path))
    
    return blur_detector, aesthetic_scorer, enhancer

def parse_uploaded_image(contents):
    """Parse uploaded image from base64 string"""
    if contents is None:
        return None
    
    # Extract base64 string
    header, encoded = contents.split(',')
    
    # Decode
    decoded = base64.b64decode(encoded)
    
    # Convert to numpy array
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return img

def image_to_base64(image, format='JPEG', quality=100, max_size=None):
    """Convert OpenCV image to base64 string for display"""
    # Resize if max_size is specified
    if max_size is not None:
        image = image_processor.resize_image(image.copy(), max_size=max_size)
    
    # Convert BGR to RGB for web display
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Use PIL to encode (handles RGB correctly)
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    
    # Use PNG for lossless quality, or high-quality JPEG
    if format == 'PNG':
        pil_image.save(buffer, format='PNG', compress_level=1)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{img_base64}'
    else:
        pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{img_base64}'

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Image Analyzer", className="text-center mb-4"),
            html.P("Upload an image to analyze blur, aesthetics, and lighting quality", 
                   className="text-center text-muted mb-4"),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Upload Image", className="card-title"),
                    dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select an Image')
                        ]),
                        style={
                            'width': '100%',
                            'height': '200px',
                            'lineHeight': '200px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'cursor': 'pointer',
                            'backgroundColor': '#f8f9fa'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status', className="mt-3"),
                ])
            ], className="mb-4"),
        ], width=12, md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Preview", className="card-title"),
                    html.Div(id='image-preview', className="text-center"),
                ])
            ], className="mb-4"),
        ], width=12, md=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Analyze Image", id="analyze-btn", color="primary", 
                      size="lg", className="w-100 mb-4", disabled=True),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='analysis-results'),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Enhance Image", id="enhance-btn", 
                      color="success", size="lg", className="w-100 mb-4", 
                      disabled=True),
            html.Div(id='enhance-recommendation', className="text-center text-muted small mt-2"),
        ], width=12),
    ]),
    
    html.Div(id='enhanced-image-container', className="mt-4"),
    
], fluid=True, className="py-4")

@callback(
    Output('upload-status', 'children'),
    Output('image-preview', 'children'),
    Output('analyze-btn', 'disabled'),
    Input('upload-image', 'contents')
)
def handle_upload(contents):
    """Handle image upload"""
    if contents is None:
        return "", html.Div("No image uploaded"), True
    
    try:
        img = parse_uploaded_image(contents)
        if img is None:
            return dbc.Alert("Failed to parse image", color="danger"), "", True
        
        # Resize for preview (keep smaller for initial preview)
        preview_img = image_processor.resize_image(img.copy(), max_size=600)
        img_base64 = image_to_base64(preview_img, quality=95)
        
        status = dbc.Alert(f"✓ Image uploaded successfully ({img.shape[1]}x{img.shape[0]})", 
                          color="success")
        preview = html.Img(src=img_base64, style={'maxWidth': '100%', 'height': 'auto'})
        
        return status, preview, False
    except Exception as e:
        error_msg = dbc.Alert(f"Error uploading image: {str(e)}", color="danger")
        return error_msg, "", True

@callback(
    Output('analysis-results', 'children'),
    Output('enhance-btn', 'disabled'),
    Output('enhance-recommendation', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('upload-image', 'contents'),
    prevent_initial_call=True
)
def analyze_image(n_clicks, contents):
    """Analyze uploaded image"""
    if contents is None:
        return "", True, html.P("")
    
    try:
        # Parse image
        img = parse_uploaded_image(contents)
        if img is None:
            return dbc.Alert("Failed to parse image", color="danger"), True, html.P("")
        
        # Get models
        blur_model, aesthetic_model, enhancer_model = get_models()
        
        # Analyze
        blur_score = blur_model.predict(img)
        aesthetic_scores = aesthetic_model.score(img)
        needs_enhancement, brightness_score = image_processor.check_lighting(img)
        
        # Create results cards
        results = []
        
        # Blur Score
        blur_color = "success" if blur_score >= 7 else ("warning" if blur_score >= 4 else "danger")
        blur_status = "Sharp" if blur_score >= 7 else ("Moderate" if blur_score >= 4 else "Blurry")
        results.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Blur Detection", className="card-title"),
                        html.H2(f"{blur_score:.1f}/10", className=f"text-{blur_color}"),
                        html.P(blur_status, className="text-muted"),
                        dbc.Progress(value=blur_score * 10, color=blur_color, className="mt-2")
                    ])
                ])
            ], width=12, md=4, className="mb-3")
        )
        
        # Aesthetic Scores
        overall_score = aesthetic_scores['overall']
        aesthetic_color = "success" if overall_score >= 7 else ("warning" if overall_score >= 5 else "danger")
        results.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Aesthetic Quality", className="card-title"),
                        html.H2(f"{overall_score:.1f}/10", className=f"text-{aesthetic_color}"),
                        html.Hr(),
                        html.P(f"Composition: {aesthetic_scores['composition']:.1f}/10", className="mb-1"),
                        html.P(f"Color: {aesthetic_scores['color']:.1f}/10", className="mb-1"),
                        html.P(f"Contrast: {aesthetic_scores['contrast']:.1f}/10", className="mb-1"),
                        html.P(f"Focus: {aesthetic_scores['focus']:.1f}/10", className="mb-1"),
                        dbc.Progress(value=overall_score * 10, color=aesthetic_color, className="mt-2")
                    ])
                ])
            ], width=12, md=4, className="mb-3")
        )
        
        # Lighting Score
        lighting_color = "success" if brightness_score >= 7 else ("warning" if brightness_score >= 4 else "danger")
        lighting_status = "Good" if brightness_score >= 7 else ("Moderate" if brightness_score >= 4 else "Dark")
        results.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Lighting Quality", className="card-title"),
                        html.H2(f"{brightness_score:.1f}/10", className=f"text-{lighting_color}"),
                        html.P(lighting_status, className="text-muted"),
                        html.P(f"Enhancement recommended: {'Yes' if needs_enhancement else 'No'}", 
                              className="small text-muted"),
                        dbc.Progress(value=brightness_score * 10, color=lighting_color, className="mt-2")
                    ])
                ])
            ], width=12, md=4, className="mb-3")
        )
        
        # Show enhance button with recommendation status
        if needs_enhancement:
            recommendation = html.P("Enhancement recommended for better lighting", 
                                   className="text-warning mb-0")
        else:
            recommendation = html.P("Enhancement available (optional)", 
                                   className="text-muted mb-0")
        
        return dbc.Row(results), False, recommendation
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error analyzing image: {str(e)}", color="danger")
        return error_msg, True, html.P("")

@callback(
    Output('enhanced-image-container', 'children'),
    Input('enhance-btn', 'n_clicks'),
    State('upload-image', 'contents'),
    prevent_initial_call=True
)
def enhance_image(n_clicks, contents):
    """Enhance low-light image"""
    if contents is None:
        return ""
    
    try:
        # Parse image
        img = parse_uploaded_image(contents)
        if img is None:
            return dbc.Alert("Failed to parse image", color="danger")
        
        # Get enhancer
        _, _, enhancer_model = get_models()
        
        # Enhance with color preservation (more natural, less over-saturated)
        enhanced_img = enhancer_model.enhance(img, preserve_colors=True)
        
        # Use higher resolution for comparison (1200px max, or original size if smaller)
        # Use PNG format for better quality (lossless)
        img_base64 = image_to_base64(enhanced_img, format='PNG', max_size=1200)
        original_base64 = image_to_base64(img, format='PNG', max_size=1200)
        
        return dbc.Card([
            dbc.CardBody([
                html.H5("Enhanced Image", className="card-title mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.H6("Original", className="text-center mb-3"),
                        html.Img(src=original_base64, style={
                            'maxWidth': '100%', 
                            'height': 'auto',
                            'imageRendering': 'auto'
                        }, className="img-fluid")
                    ], width=12, md=6),
                    dbc.Col([
                        html.H6("Enhanced", className="text-center mb-3"),
                        html.Img(src=img_base64, style={
                            'maxWidth': '100%', 
                            'height': 'auto',
                            'imageRendering': 'auto'
                        }, className="img-fluid")
                    ], width=12, md=6),
                ])
            ])
        ])
        
    except Exception as e:
        return dbc.Alert(f"Error enhancing image: {str(e)}", color="danger")

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Image Analyzer is starting...")
    print("="*50)
    print(f"\nOpen your browser and navigate to:")
    print(f"   http://localhost:8050")
    print(f"\nPress Ctrl+C to stop the server\n")
    print("="*50 + "\n")
    app.run_server(debug=True, host='127.0.0.1', port=8050)
