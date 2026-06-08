"""Main Dash application for factor performance dashboard"""
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import pandas as pd
from .config import Config
from .data_loader import load_factor_data, filter_factors, sort_factors, get_performance_summary
from .detail_view import create_detail_modal, create_loading_modal, encode_image_to_base64


# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Default configuration (can be overridden)
# Try to use absolute path if relative path doesn't work
import os
workspace_path = os.environ.get('WORKSPACE_PATH', '')
default_csv_path = 'records/cicso0/ims/rulex/200037/nxt1_ret_15h/final.csv'
if workspace_path:
    default_csv_path = os.path.join(workspace_path, default_csv_path)

default_config = Config(csv_path=default_csv_path)

# Global variable to store current data
current_df = None


# Initialize data on app start
def initialize_data():
    """Initialize data when app starts"""
    global current_df
    try:
        current_df = load_factor_data(default_config)
        source_options = [
            {'label': 'All', 'value': None}
        ] + [
            {'label': str(source), 'value': str(source)}
            for source in sorted(current_df['source'].unique())
        ]
        category_options = [
            {'label': 'All', 'value': None}
        ] + [
            {'label': str(cat), 'value': str(cat)}
            for cat in sorted(current_df['category'].unique())
        ]
        return current_df.to_dict('records'), source_options, category_options
    except Exception as e:
        return [], [{'label': f'Error: {str(e)}', 'value': None}], [{'label': 'All', 'value': None}]


# Store initial data (call before create_layout)
initial_data, initial_source_options, initial_category_options = initialize_data()


def create_layout():
    """Create the main layout of the dashboard"""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Factor Performance Dashboard", className="text-center mb-4"),
                html.P("Explore and analyze factor performance data", className="text-center text-muted mb-4")
            ])
        ]),
        
        # Search and Filters Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Search & Filters", className="mb-3"),
                        
                        # Search box
                        dbc.InputGroup([
                            dbc.InputGroupText("🔍"),
                            dbc.Input(
                                id="search-input",
                                placeholder="Search by formula, detail, or description...",
                                type="text"
                            )
                        ], className="mb-3"),
                        
                        # Filters
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Source:"),
                                dcc.Dropdown(
                                    id="source-filter",
                                    options=[],  # Will be populated dynamically
                                    value=None,
                                    clearable=True,
                                    placeholder="All sources"
                                )
                            ], md=2),
                            dbc.Col([
                                dbc.Label("Category:"),
                                dcc.Dropdown(
                                    id="category-filter",
                                    options=[],  # Will be populated dynamically
                                    value=None,
                                    clearable=True,
                                    placeholder="All categories"
                                )
                            ], md=2),
                            dbc.Col([
                                dbc.Label("Level:"),
                                dcc.Dropdown(
                                    id="level-filter",
                                    options=[
                                        {'label': 'All', 'value': None},
                                        {'label': '1', 'value': 1},
                                        {'label': '2', 'value': 2},
                                        {'label': '3', 'value': 3},
                                        {'label': '4', 'value': 4},
                                        {'label': '5', 'value': 5},
                                    ],
                                    value=None,
                                    clearable=False
                                )
                            ], md=2),
                            dbc.Col([
                                dbc.Label("Direction:"),
                                dcc.Dropdown(
                                    id="direction-filter",
                                    options=[
                                        {'label': 'All', 'value': None},
                                        {'label': 'Up (1)', 'value': 1},
                                        {'label': 'Down (-1)', 'value': -1},
                                    ],
                                    value=None,
                                    clearable=False
                                )
                            ], md=2),
                            dbc.Col([
                                dbc.Label("Score Range:"),
                                dbc.InputGroup([
                                    dbc.Input(id="min-score", type="number", placeholder="Min", step=0.1),
                                    dbc.Input(id="max-score", type="number", placeholder="Max", step=0.1)
                                ], size="sm")
                            ], md=2)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Clear Filters", id="clear-filters-btn", color="secondary", size="sm", className="float-end")
                            ])
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Data Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id="factors-table-container")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Detail Modal with Loading
        dcc.Loading(
            id="detail-modal-loading",
            type="circle",
            children=html.Div(id="detail-modal-container"),
            style={"minHeight": "200px"}
        ),
        
        # Store for current data
        dcc.Store(id="current-data-store", data=initial_data),
        
        # Store for factor data to load (trigger for data loading)
        dcc.Store(id="factor-load-store", data=None),
        
        # Source filter options (set initially)
        html.Div(id="source-filter-store", style={"display": "none"}),
        
        # Hidden div to trigger initial load
        html.Div(id="init-trigger", style={"display": "none"}, children="init")
    ], fluid=True)


app.layout = create_layout()


@app.callback(
    [Output("current-data-store", "data"),
     Output("source-filter", "options"),
     Output("category-filter", "options")],
    Input("init-trigger", "children"),
    prevent_initial_call=False
)
def load_initial_data(_):
    """Load initial data and populate source and category filters"""
    return initial_data, initial_source_options, initial_category_options


@app.callback(
    Output("factors-table-container", "children"),
    [Input("current-data-store", "data"),
     Input("search-input", "value"),
     Input("source-filter", "value"),
     Input("category-filter", "value"),
     Input("level-filter", "value"),
     Input("direction-filter", "value"),
     Input("min-score", "value"),
     Input("max-score", "value")],
    prevent_initial_call=False
)
def update_table(data_store, search_term, source, category, level, direction,
                 min_score, max_score):
    """Update the factors table based on filters"""
    if not data_store:
        return html.Div("No data available")
    
    try:
        df = pd.DataFrame(data_store)
        
        # Apply filters
        filtered_df = filter_factors(
            df,
            search_term=search_term,
            category=category,
            min_score=min_score,
            max_score=max_score,
            level=level,
            direction=direction,
            source=source,
            has_image=None
        )
        
        # Sort by score descending
        filtered_df = sort_factors(filtered_df, sort_by='score', ascending=False)
        
        # Prepare table data
        table_df = filtered_df.copy()
        table_df['formula_short'] = table_df['formula'].apply(
            lambda x: x[:60] + '...' if len(str(x)) > 60 else str(x)
        )
        
        # Create table - store row indices for detail view
        table_df['row_index'] = table_df.index
        
        # Create table
        columns = [
            {'name': 'Formula', 'id': 'formula_short'},
            {'name': 'Source', 'id': 'source'},
            {'name': 'Level', 'id': 'level'},
            {'name': 'Score', 'id': 'score', 'type': 'numeric', 'format': {'specifier': '.2f'}},
            {'name': 'Direction', 'id': 'direction'}
        ]
        
        # Format table data
        table_data = []
        for i, (_, row) in enumerate(table_df.iterrows()):
            formula_full = str(row['formula'])
            formula_short = str(row['formula_short'])
            
            table_data.append({
                'formula_short': formula_short,
                'source': str(row['source']),
                'level': int(row['level']),
                'score': float(row['score']),
                'direction': int(row['direction']),
                '_row_index': int(row['row_index'])  # Store index for click handling
            })
        
        # Store row indices for detail view - map table row to original dataframe index
        table_data_with_idx = table_data.copy()  # Already has _row_index set correctly
        
        # Generate style conditions based on score values
        # Rules: Score > 7: 绿色, Score = 7: 蓝色, Score = 6: 白色, Score > 6 and < 7: 红色, Score < 6: 红色
        # Using filter_query with proper ordering (more specific conditions first)
        style_conditions = [
            {
                'if': {'filter_query': '{score} > 7'},
                'backgroundColor': '#d4edda',  # 绿色 - 7分以上
            },
            {
                'if': {'filter_query': '{score} >= 6.99 && {score} <= 7.01'},
                'backgroundColor': '#cfe2ff',  # 蓝色 - 7分
            },
            {
                'if': {'filter_query': '{score} >= 5.99 && {score} <= 6.01'},
                'backgroundColor': '#ffffff',  # 白色 - 6分
            },
            {
                'if': {'filter_query': '{score} > 6 && {score} < 6.99'},
                'backgroundColor': '#f8d7da',  # 红色 - 6分以上但小于7分
            },
            {
                'if': {'filter_query': '{score} < 6'},
                'backgroundColor': '#f8d7da',  # 红色 - 小于6分
            },
            {
                'if': {'state': 'selected'},
                'backgroundColor': '#b3d9ff',
            }
        ]
        
        # Wrap table in a container with click handling
        table = html.Div([
            dash_table.DataTable(
                id="factors-table",
                columns=columns,
                data=table_data_with_idx,
                sort_action="native",
                filter_action="native",
                page_action="native",
                page_current=0,
                page_size=20,
                row_selectable="single",
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=style_conditions,
                tooltip_data=[
                    {
                        'formula_short': {'value': str(formula_full), 'type': 'markdown'}
                    }
                    for _, formula_full in table_df['formula'].items()
                ],
                tooltip_duration=None
            ),
            html.Div(id="table-click-store", style={"display": "none"})
        ])
        
        return table
    except Exception as e:
        return html.Div(f"Error: {str(e)}")


@app.callback(
    [Output("detail-modal-container", "children"),
     Output("factor-load-store", "data"),
     Output("table-click-store", "children")],
    [Input("factors-table", "active_cell")],
    [State("current-data-store", "data"),
     State("factors-table", "data"),
     State("factors-table", "page_current"),
     State("factors-table", "page_size")],
    prevent_initial_call=True
)
def show_detail_modal_loading(active_cell, data_store, table_data, page_current, page_size):
    """Show loading modal immediately when a row is clicked, and trigger data loading"""
    if not data_store or not active_cell or not table_data:
        return html.Div(), None, dash.no_update
    
    try:
        # Get the clicked row index
        clicked_row_idx = active_cell['row']
        
        # Calculate actual row index considering pagination
        actual_idx = page_current * page_size + clicked_row_idx
        
        # Get the row_index from table data (which maps to original dataframe)
        if actual_idx >= len(table_data):
            return html.Div(), None, dash.no_update
        
        original_row_idx = table_data[actual_idx]['_row_index']
        
        # Get factor data from original dataframe
        df = pd.DataFrame(data_store)
        if original_row_idx >= len(df):
            return html.Div(), None, dash.no_update
        
        factor_row = df.iloc[original_row_idx].to_dict()
        
        # Show loading modal immediately
        loading_modal = create_loading_modal(formula=str(factor_row.get('formula', 'Loading...')))
        
        # Store factor data to trigger data loading callback
        return loading_modal, factor_row, str(original_row_idx)
    except Exception as e:
        import traceback
        error_msg = f"Error showing details: {str(e)}\n{traceback.format_exc()}"
        error_modal = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody([
                    html.P("An error occurred while loading factor details:", className="text-danger"),
                    html.Pre(error_msg, style={'white-space': 'pre-wrap', 'font-size': '12px'})
                ])
            ],
            id="error-modal",
            is_open=True,
            size="lg"
        )
        return error_modal, None, dash.no_update


@app.callback(
    Output("detail-modal-container", "children", allow_duplicate=True),
    [Input("factor-load-store", "data")],
    prevent_initial_call=True
)
def load_factor_data(factor_data):
    """Load factor image and summary data, then update modal"""
    if not factor_data:
        return dash.no_update
    
    try:
        # Load image and summary (this may take some time)
        from .data_loader import get_factor_image_and_summary
        image_path, summary_path = get_factor_image_and_summary(
            config=default_config,
            category=str(factor_data.get('category', '')),
            source=str(factor_data.get('source', '')),
            formula=str(factor_data.get('formula', ''))
        )
        
        image_base64 = encode_image_to_base64(image_path) if image_path else None
        summary_text = get_performance_summary(summary_path) if summary_path else None
        
        # Create modal with actual data
        modal = create_detail_modal(factor_data, image_base64, summary_text, is_open=True)
        
        return modal
    except Exception as e:
        import traceback
        error_msg = f"Error loading factor data: {str(e)}\n{traceback.format_exc()}"
        error_modal = dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody([
                    html.P("An error occurred while loading factor details:", className="text-danger"),
                    html.Pre(error_msg, style={'white-space': 'pre-wrap', 'font-size': '12px'})
                ])
            ],
            id="error-modal",
            is_open=True,
            size="lg"
        )
        return error_modal


@app.callback(
    [Output("search-input", "value"),
     Output("source-filter", "value"),
     Output("category-filter", "value"),
     Output("level-filter", "value"),
     Output("direction-filter", "value"),
     Output("min-score", "value"),
     Output("max-score", "value")],
    Input("clear-filters-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_filters(n_clicks):
    """Clear all filters"""
    if n_clicks:
        return None, None, None, None, None, None, None
    return [dash.no_update] * 7


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)

