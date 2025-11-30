"""
Dash Frontend for Image Analyzer
"""
import sys
import os
from pathlib import Path

# Clean sys.path to ensure we only use packages from the conda environment
# Remove any problematic custom package paths that might interfere
problematic_paths = [r'C:\py_pkgs']
for path in problematic_paths:
    if path in sys.path:
        sys.path.remove(path)

# Also check PYTHONPATH environment variable
if 'PYTHONPATH' in os.environ:
    pythonpath = os.environ['PYTHONPATH']
    if r'C:\py_pkgs' in pythonpath:
        # Remove it from PYTHONPATH
        paths = pythonpath.split(os.pathsep)
        paths = [p for p in paths if r'C:\py_pkgs' not in p]
        os.environ['PYTHONPATH'] = os.pathsep.join(paths)

# Add project root to Python path so we can import models, data, etc.
# Get the project root (parent of scripts directory)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
from models.lighting_assessor import LightingAssessor
from utils.image_processing import ImageProcessor

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Image Analyzer"

# Initialize models (lazy loading)
blur_detector = None
aesthetic_scorer = None
enhancer = None
lighting_assessor = None
image_processor = ImageProcessor()

def get_models():
    """Lazy load models on first use"""
    global blur_detector, aesthetic_scorer, enhancer, lighting_assessor
    
    if blur_detector is None:
        blur_detector = BlurDetector()
        # Try to load weights if available
        weights_path = Path('outputs/weights/blur_detector_best.pth')
        if weights_path.exists():
            blur_detector.load_weights(str(weights_path))
    
    if aesthetic_scorer is None:
        aesthetic_scorer = AestheticScorer()
        weights_path = Path('outputs/weights/aesthetic_scorer_best.pth')
        if weights_path.exists():
            aesthetic_scorer.load_weights(str(weights_path))
    
    if enhancer is None:
        # Try curve network first (more realistic), fallback to U-Net if weights not found
        enhancer = None
        curve_weights = Path('outputs/weights/enhancer_curve_best.pth')
        unet_weights = Path('outputs/weights/enhancer_best.pth')
        
        # Try curve network first
        if curve_weights.exists():
            enhancer = LowLightEnhancer(architecture='curve')
            enhancer.load_weights(str(curve_weights))
        # Fallback to U-Net if curve weights don't exist
        elif unet_weights.exists():
            enhancer = LowLightEnhancer(architecture='unet')
            enhancer.load_weights(str(unet_weights))
        else:
            # No weights found, use curve network by default (better architecture)
            enhancer = LowLightEnhancer(architecture='curve')
            print("⚠ No trained weights found. Using CV-only enhancement.")
    
    if lighting_assessor is None:
        lighting_assessor = LightingAssessor()
        weights_path = Path('outputs/weights/lighting_assessor_best.pth')
        if weights_path.exists():
            lighting_assessor.load_weights(str(weights_path))
    
    return blur_detector, aesthetic_scorer, enhancer, lighting_assessor

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
            html.Div(id='model-status', className="mb-3"),
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
            html.Div([
                html.Label("Max Brightness Limit (0-255):", className="form-label mt-3"),
                dcc.Slider(
                    id='brightness-slider',
                    min=180,
                    max=255,
                    step=5,
                    value=220,
                    marks={180: '180', 200: '200', 220: '220', 240: '240', 255: '255'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.P("Lower values = less bright enhancement", className="text-muted small mt-2"),
            ], className="mb-3"),
            # Enhancement mode toggle buttons
            html.Div([
                html.Label("Enhancement Mode:", className="form-label mt-3 mb-2"),
                dbc.ButtonGroup([
                    dbc.Button("CV Only", id="mode-cv-btn", color="primary", outline=False, className="me-2"),
                    dbc.Button("CV + ML", id="mode-hybrid-btn", color="secondary", outline=True),
                ], className="w-100 mb-2"),
                html.Div(id='mode-description', className="text-muted small mt-2"),
            ], className="mb-3"),
            # ML strength slider (only shown when CV+ML is selected)
            html.Div([
                html.Label("ML Refinement Strength:", className="form-label mt-2 mb-2", id='ml-strength-label'),
                dcc.Slider(
                    id='ml-strength-slider',
                    min=0.01,
                    max=0.3,
                    step=0.01,
                    value=0.05,
                    marks={0.01: '1%', 0.05: '5%', 0.1: '10%', 0.2: '20%', 0.3: '30%'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    disabled=True
                ),
                html.P("Higher values = more ML influence (may cause artifacts)", className="text-muted small mt-2", id='ml-strength-help'),
            ], className="mb-3", id='ml-strength-container', style={'display': 'none'}),
            # Store for enhancement mode
            dcc.Store(id='enhancement-mode-store', data=False),  # False = CV only, True = CV+ML
        ], width=12),
    ]),
    
    html.Div(id='enhanced-image-container', className="mt-4"),
    html.Div(id='enhancement-toast', className="mt-3"),
    
], fluid=True, className="py-4")

@callback(
    Output('model-status', 'children'),
    Input('upload-image', 'contents'),
    prevent_initial_call=False
)
def update_model_status(contents):
    """Display model status (ML vs fallback)"""
    weights_dir = Path('outputs/weights')
    models_status = []
    
    # Check which models have weights
    has_blur = (weights_dir / 'blur_detector_best.pth').exists()
    has_aesthetic = (weights_dir / 'aesthetic_scorer_best.pth').exists()
    has_enhancer = (weights_dir / 'enhancer_best.pth').exists()
    has_lighting = (weights_dir / 'lighting_assessor_best.pth').exists()
    
    ml_count = sum([has_blur, has_aesthetic, has_enhancer, has_lighting])
    
    if ml_count == 0:
        return dbc.Alert([
            html.Strong("⚠ Using Fallback Methods: "),
            "No trained model weights found. The system is using classical computer vision methods. ",
            "To use ML models, train them using the training scripts (see TRAINING_GUIDE.md or QUICK_START.md)."
        ], color="warning", className="mb-3")
    elif ml_count < 4:
        return dbc.Alert([
            html.Strong("ℹ Partial ML Mode: "),
            f"{ml_count}/4 models using ML. Some features may use fallback methods."
        ], color="info", className="mb-3")
    else:
        return dbc.Alert([
            html.Strong("✓ ML Models Active: "),
            "All models are using trained ML weights."
        ], color="success", className="mb-3")

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
        blur_model, aesthetic_model, enhancer_model, lighting_model = get_models()
        
        # Analyze
        blur_score = blur_model.predict(img)
        aesthetic_scores = aesthetic_model.score(img)
        
        # Use ML-based lighting assessment if available, otherwise fallback
        try:
            brightness_score, needs_enhancement = lighting_model.assess(img)
        except Exception as e:
            # Fallback to classical method if ML model fails
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

# Enhancement mode toggle callbacks
@callback(
    Output('enhancement-mode-store', 'data'),
    Input('mode-cv-btn', 'n_clicks'),
    Input('mode-hybrid-btn', 'n_clicks'),
    State('enhancement-mode-store', 'data'),
    prevent_initial_call=False
)
def update_enhancement_mode(n_cv, n_hybrid, current_mode):
    """Update enhancement mode based on button clicks"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return False  # Default to CV only
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'mode-cv-btn':
        return False  # CV only
    elif button_id == 'mode-hybrid-btn':
        return True  # CV + ML
    return current_mode if current_mode is not None else False

@callback(
    Output('mode-cv-btn', 'color'),
    Output('mode-cv-btn', 'outline'),
    Output('mode-hybrid-btn', 'color'),
    Output('mode-hybrid-btn', 'outline'),
    Output('mode-description', 'children'),
    Output('ml-strength-container', 'style'),
    Output('ml-strength-slider', 'disabled'),
    Input('enhancement-mode-store', 'data'),
    prevent_initial_call=False
)
def update_mode_buttons(use_hybrid):
    """Update button styles and description based on selected mode"""
    if use_hybrid:
        return (
            "secondary", True,  # CV button: secondary, outlined
            "primary", False,   # Hybrid button: primary, filled
            "CV preprocessing + ML refinement - Experimental",
            {'display': 'block'},  # Show ML strength slider
            False  # Enable ML strength slider
        )
    else:
        return (
            "primary", False,   # CV button: primary, filled
            "secondary", True,  # Hybrid button: secondary, outlined
            "Classical CV only (CLAHE + gamma) - Recommended for best quality",
            {'display': 'none'},  # Hide ML strength slider
            True  # Disable ML strength slider
        )

@callback(
    Output('enhanced-image-container', 'children'),
    Output('enhancement-toast', 'children'),
    Input('enhance-btn', 'n_clicks'),
    State('upload-image', 'contents'),
    State('brightness-slider', 'value'),
    State('enhancement-mode-store', 'data'),
    State('ml-strength-slider', 'value'),
    prevent_initial_call=True
)
def enhance_image(n_clicks, contents, max_brightness, use_hybrid_mode, ml_strength):
    """Enhance low-light image"""
    if contents is None:
        return "", ""
    
    try:
        # Parse image
        img = parse_uploaded_image(contents)
        if img is None:
            return dbc.Alert("Failed to parse image", color="danger"), ""
        
        # Get enhancer
        _, _, enhancer_model, _ = get_models()
        
        # Get enhancement mode from store
        use_hybrid = use_hybrid_mode if use_hybrid_mode is not None else False
        ml_strength = ml_strength if ml_strength is not None else 0.05
        
        # Enhance with color preservation and brightness limiting (prevent over-enhancement)
        # max_brightness prevents images from becoming too bright (0-255 scale)
        enhanced_img = enhancer_model.enhance(
            img, 
            preserve_colors=True, 
            max_brightness=max_brightness,
            use_hybrid=use_hybrid,
            ml_strength=ml_strength
        )
        
        # Use higher resolution for comparison (1200px max, or original size if smaller)
        # Use PNG format for better quality (lossless)
        img_base64 = image_to_base64(enhanced_img, format='PNG', max_size=1200)
        original_base64 = image_to_base64(img, format='PNG', max_size=1200)
        
        # Get method used from enhancer
        method_used = getattr(enhancer_model, 'last_method_used', 'CV')
        
        # Create toast notification
        if method_used == "CV+ML":
            toast = dbc.Toast(
                f"Enhanced using CV preprocessing + ML refinement ({ml_strength*100:.0f}% ML strength)",
                header="Enhancement Complete",
                icon="success",
                dismissable=True,
                is_open=True,
                duration=4000,
                style={"position": "fixed", "top": 66, "right": 10, "width": 350},
            )
        elif method_used == "ML":
            toast = dbc.Toast(
                "Enhanced using ML model only",
                header="Enhancement Complete",
                icon="success",
                dismissable=True,
                is_open=True,
                duration=4000,
                style={"position": "fixed", "top": 66, "right": 10, "width": 350},
            )
        else:
            toast = dbc.Toast(
                "Enhanced using classical CV methods (CLAHE + gamma correction)",
                header="Enhancement Complete",
                icon="info",
                dismissable=True,
                is_open=True,
                duration=4000,
                style={"position": "fixed", "top": 66, "right": 10, "width": 350},
            )
        
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
        ]), toast
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error enhancing image: {str(e)}", color="danger")
        return error_msg, ""

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Image Analyzer is starting...")
    print("="*50)
    print(f"\nOpen your browser and navigate to:")
    print(f"   http://localhost:8050")
    print(f"\nPress Ctrl+C to stop the server\n")
    print("="*50 + "\n")
    app.run_server(debug=False, host='127.0.0.1', port=8050)
