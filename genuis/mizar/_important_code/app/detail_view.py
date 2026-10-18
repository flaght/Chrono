"""Detail view component for factor performance"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import Dict, Optional
import base64


def create_detail_modal(factor_data: Dict, image_base64: Optional[str] = None,
                        summary_text: Optional[str] = None, is_open: bool = True) -> dbc.Modal:
    """
    Create a modal dialog showing factor details.
    
    Args:
        factor_data: Dictionary containing factor information
        image_base64: Base64-encoded image data
        summary_text: Performance summary text
        is_open: Whether the modal is open
    
    Returns:
        Dash Bootstrap Modal component
    """
    # Extract factor information
    formula = factor_data.get('formula', 'N/A')
    direction = factor_data.get('direction', 'N/A')
    source = factor_data.get('source', 'N/A')
    level = factor_data.get('level', 'N/A')
    category = factor_data.get('category', 'N/A')
    score = factor_data.get('score', 'N/A')
    detail = factor_data.get('detail', 'N/A')
    desc = factor_data.get('desc', 'N/A')
    factor_id = factor_data.get('factor_id', 'N/A')
    
    # Build modal content
    modal_content = [
        dbc.ModalHeader(dbc.ModalTitle(f"Factor Details - ID: {factor_id}")),
        dbc.ModalBody([
            # Basic Information Section
            html.H5("Basic Information", className="mb-3"),
            dbc.Table([
                html.Tbody([
                    html.Tr([html.Td("Formula:", className="fw-bold"), html.Td(formula)]),
                    html.Tr([html.Td("Category:", className="fw-bold"), html.Td(category)]),
                    html.Tr([html.Td("Source:", className="fw-bold"), html.Td(str(source))]),
                    html.Tr([html.Td("Level:", className="fw-bold"), html.Td(str(level))]),
                    html.Tr([html.Td("Score:", className="fw-bold"), html.Td(f"{score:.2f}" if isinstance(score, (int, float)) else str(score))]),
                    html.Tr([html.Td("Direction:", className="fw-bold"), html.Td(str(direction))]),
                    html.Tr([html.Td("Factor ID:", className="fw-bold"), html.Td(str(factor_id))]),
                ])
            ], bordered=True, hover=True, className="mb-4"),
            
            # Description Section
            html.H5("Description", className="mb-3"),
            html.Div([
                html.P(html.Strong("Detail:"), className="mb-2"),
                html.P(detail if detail != 'N/A' else 'No detail available', className="text-muted mb-3"),
                html.P(html.Strong("Desc:"), className="mb-2"),
                html.P(desc if desc != 'N/A' else 'No description available', className="text-muted"),
            ], className="mb-4"),
            
            # Performance Image Section
            html.H5("Performance Plot", className="mb-3"),
            html.Div([
                html.Img(
                    src=image_base64 if image_base64 else '/assets/placeholder.png',
                    style={
                        'width': '100%',
                        'max-width': '1000px',
                        'height': 'auto',
                        'border': '1px solid #ddd',
                        'border-radius': '4px'
                    },
                    className="mb-3"
                ) if image_base64 else html.Div([
                    html.I(className="fas fa-image fa-3x text-muted mb-2"),
                    html.P("Performance plot not available", className="text-muted")
                ], className="text-center py-4")
            ], className="mb-4"),
            
            # Performance Summary Section
            html.H5("Performance Summary", className="mb-3"),
            html.Div([
                html.Pre(
                    summary_text if summary_text else 'Performance summary not available',
                    style={
                        'white-space': 'pre-wrap',
                        'font-family': 'monospace',
                        'font-size': '12px',
                        'background-color': '#f8f9fa',
                        'padding': '15px',
                        'border-radius': '4px',
                        'max-height': '400px',
                        'overflow-y': 'auto'
                    }
                )
            ])
        ])
    ]
    
    return dbc.Modal(
        modal_content,
        id="factor-detail-modal",
        is_open=is_open,
        size="xl",
        scrollable=True
    )


def create_loading_modal(formula: str = "Loading...") -> dbc.Modal:
    """
    Create a loading modal dialog that shows while factor details are being loaded.
    
    Args:
        formula: Factor formula to display in the loading message
    
    Returns:
        Dash Bootstrap Modal component with loading state
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Loading Factor Details...")),
            dbc.ModalBody([
                html.Div([
                    html.Div(
                        dbc.Spinner(
                            html.Div(id="loading-spinner"),
                            size="lg",
                            color="primary"
                        ),
                        className="mb-3"
                    ),
                    html.P(f"Loading details for factor:", className="text-center mb-2"),
                    html.P(
                        formula[:100] + "..." if len(formula) > 100 else formula,
                        className="text-muted text-center",
                        style={'font-family': 'monospace', 'font-size': '12px'}
                    ),
                    html.P("Please wait...", className="text-center text-muted mt-3")
                ], className="text-center py-5")
            ])
        ],
        id="loading-modal",
        is_open=True,
        size="md",
        backdrop="static",  # Prevent closing while loading
        keyboard=False
    )


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Base64 encoded string (data URI) or None if file doesn't exist
    """
    import os
    if not os.path.exists(image_path):
        return None
    
    try:
        with open(image_path, 'rb') as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            # Determine image type from extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif'
            }.get(ext, 'image/png')
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

