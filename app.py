import dash
from dash import dcc, html
from dash.dependencies import Input, Output # State not used yet, but good to have
import os
from dotenv import load_dotenv
import sys
import pandas as pd
# import plotly.express as px # Not strictly needed if using go.Scatter directly for hist. graph
import plotly.graph_objects as go
import traceback
import numpy as np

# --- Add Project Root to sys.path consistently ---
# This assumes app.py is in your BREATHEEASY/ project root directory.
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import backend functions and variables ---
try:
    from src.api_integration.weather_client import get_current_weather
    from src.analysis.historical import get_city_aqi_trend_data
    from src.health_rules.info import AQI_DEFINITION, AQI_SCALE
    from src.exceptions import APIError
except ImportError as e:
    # This block is crucial for the app to run even if backend modules are missing/have issues.
    print(f"CRITICAL ERROR importing backend modules: {e}")
    print("Ensure 'src' directory and all its submodules with __init__.py files are present and correct.")
    traceback.print_exc() # Print full traceback for import errors

    # Define dummy functions/variables so the app can attempt to load
    def get_current_weather(city_name):
        print(f"Using DUMMY get_current_weather for {city_name}")
        return {"temp_c": "N/A", "condition_icon": None, "condition_text": "N/A", 
                "humidity": "N/A", "wind_kph": "N/A", "wind_dir": "", 
                "feelslike_c": "N/A", "pressure_mb": "N/A", "uv_index": "N/A",
                "error_message": "Backend weather client not loaded"} # Add error_message for clarity

    def get_city_aqi_trend_data(city_name):
        print(f"Using DUMMY get_city_aqi_trend_data for {city_name}")
        # Return an empty series or one with minimal data for placeholder
        return pd.Series(dtype='float64', name="AQI") 

    AQI_DEFINITION = "AQI definition not loaded (dummy). Check imports."
    AQI_SCALE = [{'level': 'Error', 'range': 'N/A', 'color': '#CCCCCC', 
                  'implications': 'AQI Scale not loaded (dummy). Check imports.'}]
    
    class APIError(Exception): pass # Define a basic APIError if the real one isn't imported

# --- Load environment variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print("Warning: .env file not found. API calls might fail or use defaults.")

# --- Hardcoded City List for Dropdown ---
HARDCODED_CITIES = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad']

# --- Initialize Dash App ---
app = dash.Dash(__name__, assets_folder='assets')
app.title = "BreatheEasy"

# --- App Layout ---
app.layout = html.Div(className="app-shell", children=[
    html.Div(className="page-header", children=[
        html.Div("LOGO", className="logo-placeholder")
    ]),

    html.Div(className="control-bar", children=[
        html.Div(className="city-dropdown-container", children=[
            dcc.Dropdown(
                id='city-dropdown',
                options=[{'label': city, 'value': city} for city in HARDCODED_CITIES],
                value=HARDCODED_CITIES[0] if HARDCODED_CITIES else None,
                placeholder="Select a city",
                clearable=False,
                className="city-dropdown"
            )
        ]),
        html.Div(id='current-weather-display', className="current-weather-display")
    ]),

    html.Div(className="main-content-grid", children=[
        # Row 1
        html.Div(className="widget-card", id="section-1-hist-summary", children=[
            html.H3("Section 1: Historical Summary"),
            dcc.Graph(
                id='historical-aqi-trend-graph',
                figure={}, 
                config={'responsive': True, 'displayModeBar': False},
                className='flex-graph-container' # Styled by CSS to grow
            )
        ]),
        html.Div(className="widget-card", id="section-3-curr-aqi", children=[html.H3("Section 3: Current AQI"), html.P("Content for Current AQI...")]),
        html.Div(className="widget-card", id="section-5-pollutant-risks", children=[html.H3("Section 5: Current Pollutant Risks"), html.P("Content for Pollutant Risks...")]),

        # Row 2
        html.Div(className="widget-card", id="section-2-edu-info", children=[
            html.H3("Section 2: AQI Educational Info"),
            html.Div(className="edu-info-content", children=[
                dcc.Markdown(
                    AQI_DEFINITION,
                    className="aqi-definition-markdown"
                ),
                html.Hr(className="edu-info-separator"),
                html.H4("AQI Categories (CPCB India)", className="aqi-scale-title"),
                html.Div(className="aqi-scale-container", children=[
                    html.Div(
                        className="aqi-category-card",
                        style={'borderColor': category['color'], 'backgroundColor': f"{category['color']}20"}, 
                        children=[
                            html.Strong(f"{category['level']} ", className="aqi-category-level"),
                            html.Span(f"({category['range']})", className="aqi-category-range"),
                            html.P(category['implications'], className="aqi-category-implications")
                        ]
                    ) for category in AQI_SCALE
                ])
            ])
        ]),
        html.Div(className="widget-card", id="section-4-aqi-forecast", children=[html.H3("Section 4: AQI Forecast"), html.P("Content for AQI Forecast...")]),
        html.Div(className="widget-card", id="section-6-weekly-risks", children=[html.H3("Section 6: Predicted Weekly Risks"), html.P("Content for Weekly Risks...")]),
    ]),

    html.Div(className="page-footer", children=[
        html.P("Project Team: Arnav Vidya, Chirag P Patil, Kimaya Anand | School: Delhi Public School Bangalore South"),
        html.P("Copyright © 2025 BreatheEasy Project Team. Licensed under MIT License.") # Slightly rephrased copyright
    ])
])

# --- Callbacks ---

@app.callback(
    Output('current-weather-display', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_current_weather(selected_city):
    def get_default_weather_layout(city_name_text="Select a city", error_message=None):
        condition_display_children = ["Condition N/A"]
        condition_style = {}
        if error_message:
            condition_display_children = [error_message]
            condition_style = {'font-style': 'normal', 'color': '#CC0000'}
        return [
            html.Div(className="weather-icon-container", children=[
                html.Img(src="", alt="Weather icon placeholder", className="weather-icon")
            ]),
            html.Div(className="weather-text-info-expanded", children=[
                html.P(city_name_text, className="weather-city"),
                html.P("-°C", className="weather-temp"),
                html.P(condition_display_children, className="weather-condition", style=condition_style),
                html.Div(className="weather-details-row", children=[
                    html.P("Humidity: - %", className="weather-details"),
                    html.P("Wind: - kph", className="weather-details")
                ])
            ])
        ]

    if not selected_city:
        return get_default_weather_layout()

    try:
        query_city_for_api = f"{selected_city}, India"
        weather_data = get_current_weather(query_city_for_api)

        if weather_data and isinstance(weather_data, dict) and 'temp_c' in weather_data:
            icon_url_path = weather_data.get('condition_icon')
            condition_text = weather_data.get('condition_text', "Not available")
            icon_url = "https:" + icon_url_path if icon_url_path and icon_url_path.startswith("//") else (icon_url_path or "")
            if not condition_text or str(condition_text).strip() == "": condition_text = "Not available"
            return [
                html.Div(className="weather-icon-container", children=[
                    html.Img(src=icon_url, alt=condition_text if condition_text != "Not available" else "Weather icon",
                             className="weather-icon", style={'display': 'block' if icon_url else 'none'})]),
                html.Div(className="weather-text-info-expanded", children=[
                    html.P(f"{selected_city}", className="weather-city"),
                    html.P(f"{weather_data.get('temp_c', '-')}°C", className="weather-temp"),
                    html.P(f"{condition_text}", className="weather-condition"),
                    html.Div(className="weather-details-row", children=[
                        html.P(f"Humidity: {weather_data.get('humidity', '-')} %", className="weather-details"),
                        html.P(f"Wind: {weather_data.get('wind_kph', '-')} kph {weather_data.get('wind_dir', '')}", className="weather-details"),
                        html.P(f"Feels like: {weather_data.get('feelslike_c', '-')}°C", className="weather-details"),
                        html.P(f"Pressure: {weather_data.get('pressure_mb', '-')} mb", className="weather-details"),
                        html.P(f"UV Index: {weather_data.get('uv_index', '-')}", className="weather-details"),])])]
        else:
            error_msg = "Weather data not available"
            if weather_data and isinstance(weather_data, dict):
                if weather_data.get("error_message"): error_msg = weather_data.get("error_message")
                elif weather_data.get("error"):
                    error_detail = weather_data.get("error"); error_msg = error_detail.get("message", "Unknown API error") if isinstance(error_detail, dict) else str(error_detail)
            return get_default_weather_layout(city_name_text=selected_city, error_message=error_msg)
    except APIError as e:
        print(f"Handled APIError for {selected_city}: {e}") # Dev console
        # log.error(f"APIError for weather in {selected_city}: {e}") # Production logging
        return get_default_weather_layout(city_name_text=selected_city, error_message="Weather service unavailable.")
    except Exception as e:
        print(f"General error fetching weather for {selected_city}: {e}") # Dev console
        traceback.print_exc() # Dev console
        # log.exception(f"General error for weather in {selected_city}") # Production logging
        return get_default_weather_layout(city_name_text=selected_city, error_message="Error loading weather.")
    
@app.callback(
    Output('historical-aqi-trend-graph', 'figure'),
    [Input('city-dropdown', 'value')]
)
def update_historical_trend_graph(selected_city):
    def create_placeholder_figure(message_text, height=300):
        fig = go.Figure()
        fig.update_layout(annotations=[dict(text=message_text, showarrow=False, font=dict(size=14, color="#0A4D68"))],
                          xaxis_visible=False, yaxis_visible=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=height)
        return fig

    if not selected_city: return create_placeholder_figure("Select a city to view historical AQI trend")

    try:
        aqi_trend_series = get_city_aqi_trend_data(selected_city)
        if aqi_trend_series is None or aqi_trend_series.empty:
            # log.warning(f"No historical data for {selected_city}")
            return create_placeholder_figure(f"No historical data available for {selected_city}")
        
        df_trend = aqi_trend_series.reset_index()
        if not ({'Date', 'AQI'}.issubset(df_trend.columns)): # Check if both columns exist
            if len(df_trend.columns) == 2: df_trend.columns = ['Date', 'AQI']
            else: raise ValueError(f"DataFrame columns {df_trend.columns.tolist()} not as expected ('Date', 'AQI').")

        df_trend['Date'] = pd.to_datetime(df_trend['Date'], errors='coerce')
        df_trend['AQI'] = pd.to_numeric(df_trend['AQI'], errors='coerce')
        if np.isinf(df_trend['AQI']).any(): df_trend.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_trend.dropna(subset=['Date', 'AQI'], inplace=True)
        df_trend = df_trend.sort_values(by='Date').reset_index(drop=True)
        
        if df_trend.empty:
            # log.warning(f"Data empty after cleaning for {selected_city}")
            return create_placeholder_figure(f"No valid data after cleaning for {selected_city}")

        actual_aqi_min, actual_aqi_max = df_trend['AQI'].min(), df_trend['AQI'].max()
        x_values_for_plot, y_values_for_plot = df_trend['Date'].tolist(), df_trend['AQI'].tolist()

        fig = go.Figure(data=[go.Scatter(x=x_values_for_plot, y=y_values_for_plot, mode='lines', name='AQI Trend', line_shape='linear')])
        
        yaxis_buffer = 0.1 * (actual_aqi_max - actual_aqi_min) if (actual_aqi_max - actual_aqi_min) > 0 else 10
        yaxis_min_range, yaxis_max_range = max(0, actual_aqi_min - yaxis_buffer), actual_aqi_max + yaxis_buffer
        
        graph_title = f"AQI Trend for {selected_city}" 

        fig.update_layout(title_text=graph_title, title_x=0.5, title_font_size=14, # Reduced title font slightly
                          height=380, # Adjusted height to better fit typical widget card after H3
                          xaxis_title="Date", yaxis_title="AQI Value", yaxis_range=[yaxis_min_range, yaxis_max_range],
                          margin=dict(l=50, r=20, t=40, b=40), # Adjusted margins
                          plot_bgcolor='rgba(255,255,255,0.8)', paper_bgcolor='rgba(0,0,0,0)', font_color="#0A4D68")
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightSteelBlue', tickfont_size=10) # Smaller tick font
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightSteelBlue', tickfont_size=10) # Smaller tick font
        fig.update_traces(line=dict(color='#007bff', width=1.5), hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>AQI</b>: %{y:.0f}<extra></extra>")
        
        return fig
    except Exception as e:
        print(f"ERROR IN HISTORICAL TREND CALLBACK for {selected_city}:") # Dev console
        traceback.print_exc() # Dev console
        # log.exception(f"Error generating historical trend graph for {selected_city}") # Production
        return create_placeholder_figure(f"Error displaying trend for {selected_city}", height=380)

# --- Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True)