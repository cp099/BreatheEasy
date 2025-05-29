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
import math
import dash_svg

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
    from src.api_integration.client import get_current_aqi_for_city
    from src.health_rules.info import get_aqi_info 
    from src.modeling.predictor import generate_forecast, format_forecast_for_ui
    from src.api_integration.client import get_current_pollutant_risks_for_city
    from src.exceptions import APIError, ModelFileNotFoundError
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
    
    def get_current_aqi_for_city(city_name):
        print(f"Using DUMMY get_current_aqi_for_city for {city_name}")
        # Simulate a successful call and an "Unknown station" or error
        if city_name == "Delhi, India": # Ensure you use the suffixed name if testing dummy
            return {'city': 'Delhi', 'aqi': 55, 'station': 'Dummy Station, Delhi', 'time': '2023-10-27 10:00:00'}
        elif city_name == "Mumbai, India":
            return {'city': 'Mumbai', 'aqi': 155, 'station': 'Dummy Station, Mumbai', 'time': '2023-10-27 10:05:00'}
        else:
            return {'city': city_name.split(',')[0], 'aqi': None, 'station': 'Unknown station', 'time': None, 'error': 'Station not found or API error'}

    def get_aqi_info(aqi_value):
        print(f"Using DUMMY get_aqi_info for AQI: {aqi_value}")
        if aqi_value is None: return {'level': 'N/A', 'range': '-', 'color': '#DDDDDD', 'implications': 'AQI value not available.'}
        if aqi_value <= 50: return {'level': 'Good', 'range': '0-50', 'color': '#A8E05F', 'implications': 'Minimal impact.'} # Greenish
        if aqi_value <= 100: return {'level': 'Satisfactory', 'range': '51-100', 'color': '#D4E46A', 'implications': 'Minor breathing discomfort.'} # Lighter Green/Yellow
        if aqi_value <= 200: return {'level': 'Moderate', 'range': '101-200', 'color': '#FDD74B', 'implications': 'Breathing discomfort.'} # Yellow
        return {'level': 'Poor', 'range': '201+', 'color': '#FFA500', 'implications': 'Significant breathing discomfort.'} # Orange (simplified dummy)

    def generate_forecast(target_city, days_ahead, apply_residual_correction):
        print(f"Using DUMMY generate_forecast for {target_city}")
        dates = pd.to_datetime([pd.Timestamp.now().date() + pd.Timedelta(days=i) for i in range(days_ahead)])
        return pd.DataFrame({'ds': dates, 'yhat_adjusted': [50 + i*10 for i in range(days_ahead)]})

    def format_forecast_for_ui(forecast_df):
        print("Using DUMMY format_forecast_for_ui")
        if forecast_df is None or forecast_df.empty: return []
        return [{'date': row['ds'].strftime('%Y-%m-%d'), 'predicted_aqi': int(row['yhat_adjusted'])} 
                for _, row in forecast_df.iterrows()]
    
    def get_current_pollutant_risks_for_city(city_name):
        print(f"Using DUMMY get_current_pollutant_risks_for_city for {city_name}")
        # Simulate a successful call and an error or no risks
        city_simple = city_name.split(',')[0]
        if city_simple == "Delhi":
            return {
                'city': 'Delhi', 
                'time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), 
                'pollutants': {'pm25': {'v': 160}, 'co': {'v': 5.0}}, 
                'risks': [
                    "PM2.5: Moderate - May cause breathing discomfort to people with lung disease such as asthma, and discomfort to people with heart disease, children and older adults.",
                    "CO: Satisfactory - Minor breathing discomfort to sensitive individuals."
                ]
            }
        elif city_simple == "Mumbai":
             return {
                'city': 'Mumbai', 
                'time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'), 
                'pollutants': {'pm10': {'v': 45}}, 
                'risks': ["No significant pollutant risks identified at this time."] # Or just an empty list
            }
        else: # Simulate an error or no data
            return {'city': city_simple, 'time': None, 'pollutants': {}, 'risks': [], 'error': 'Pollutant data unavailable for this city.'}
        
    class ModelFileNotFoundError(Exception): pass

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
       html.Div(className="widget-card", id="section-3-curr-aqi", children=[
            html.H3("Section 3: Current AQI"),
            html.Div(id='current-aqi-details-content', className='current-aqi-widget-content') # This Div will be populated by the callback
        ]),
        html.Div(className="widget-card", id="section-5-pollutant-risks", children=[
            html.H3("Section 5: Current Pollutant Risks"),
            html.Div(id='current-pollutant-risks-content', className='pollutant-risks-widget-content') # Populated by callback
        ]),

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
        html.Div(className="widget-card", id="section-4-aqi-forecast", children=[
            html.H3("Section 4: AQI Forecast (Next 3 Days)"),
            html.Div(id='aqi-forecast-table-content', className='forecast-widget-content')
        ]),
        html.Div(className="widget-card", id="section-6-weekly-risks", children=[html.H3("Section 6: Predicted Weekly Risks"), html.P("Content for Weekly Risks...")]),
    ]),

    html.Div(className="page-footer", children=[
        html.P("Project Team: Arnav Vidya, Chirag P Patil, Kimaya Anand | School: Delhi Public School Bangalore South"),
        html.P("Copyright © 2025 BreatheEasy Project Team. Licensed under MIT License.") # Slightly rephrased copyright
    ])
])

# --- Callbacks ---

# --- Section 1: Current Weather ---
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
    
# --- Section 2: Historical AQI Trend Graph ---
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
    
# --- Section 3: Current AQI Details ---

# --- HELPER FUNCTION FOR SVG ARC (Keep this definition in your app.py, typically above callbacks) ---
def describe_arc(x, y, radius, start_angle_deg, end_angle_deg):
    start_rad = math.radians(start_angle_deg)
    end_rad = math.radians(end_angle_deg)
    start_x = x + radius * math.cos(start_rad)
    start_y = y + radius * math.sin(start_rad)
    end_x = x + radius * math.cos(end_rad)
    end_y = y + radius * math.sin(end_rad)
    angle_diff = end_angle_deg - start_angle_deg
    if angle_diff < 0: angle_diff += 360
    large_arc_flag = "1" if angle_diff > 180 else "0"
    sweep_flag = "1"
    d = f"M {start_x} {start_y} A {radius} {radius} 0 {large_arc_flag} {sweep_flag} {end_x} {end_y}"
    return d

@app.callback(
    Output('current-aqi-details-content', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_current_aqi_details(selected_city): # selected_city is "Delhi", "Mumbai", etc.
    if not selected_city:
        return html.P("Select a city to view current AQI.", style={'textAlign': 'center', 'marginTop': '20px'})

    query_city_for_api = f"{selected_city}, India" 
    
    try:
        aqi_data = get_current_aqi_for_city(query_city_for_api)

        if not aqi_data or aqi_data.get('aqi') is None or 'error' in aqi_data:
            error_message = "Data unavailable" 
            if isinstance(aqi_data, dict) and 'error' in aqi_data:
                error_message = aqi_data['error']
            if isinstance(aqi_data, dict) and aqi_data.get('station') == "Unknown station" and "not found" in error_message.lower():
                 error_message = f"No AQI monitoring station found for {selected_city} via AQICN."
            
            return html.Div([
                html.P(f"Could not retrieve AQI for {selected_city}.", className="aqi-error-message"),
                html.P(error_message, className="aqi-error-detail")
            ], className="current-aqi-error-container")

        aqi_value = aqi_data.get('aqi')
        obs_time_str = aqi_data.get('time', 'N/A')
        
        category_info = get_aqi_info(aqi_value) 
        aqi_level = category_info.get('level', 'N/A')
        aqi_color = category_info.get('color', '#DDDDDD')
        
        formatted_time = obs_time_str
        if obs_time_str and isinstance(obs_time_str, str) and obs_time_str != 'N/A':
            try:
                dt_obj = pd.to_datetime(obs_time_str) 
                formatted_time = dt_obj.strftime('%I:%M %p, %b %d')
            except ValueError:
                if len(obs_time_str) > 30: formatted_time = "Time N/A" 
                else: formatted_time = obs_time_str
        elif hasattr(obs_time_str, 'strftime'):
             formatted_time = obs_time_str.strftime('%I:%M %p, %b %d')
        else:
            formatted_time = str(obs_time_str) if obs_time_str != 'N/A' else "Time N/A"

        # --- SVG Gauge Parameters - ADJUSTED FOR LARGER SIZE ---
        max_aqi_on_scale = 500.0
        current_aqi_clamped = max(0, min(float(aqi_value), max_aqi_on_scale)) 
        percentage = current_aqi_clamped / max_aqi_on_scale
        
        gauge_start_angle_deg = -225 
        gauge_total_sweep_deg = 270 
        value_end_angle_deg = gauge_start_angle_deg + (percentage * gauge_total_sweep_deg)

        # Increase these values significantly
        viewbox_size = 280 # Increased from 220
        center_xy = viewbox_size / 2
        radius = 115   # Increased from 90 (radius should be < center_xy - stroke_width/2)
        stroke_width = 22 # Increased from 20

        background_arc_path = describe_arc(center_xy, center_xy, radius, gauge_start_angle_deg, gauge_start_angle_deg + gauge_total_sweep_deg)
        foreground_arc_path = describe_arc(center_xy, center_xy, radius, gauge_start_angle_deg, value_end_angle_deg)
        
        return html.Div(className="aqi-gauge-wrapper", children=[
            html.H4(selected_city, className="aqi-city-name-highlight"),
            html.Div(className="aqi-gauge-svg-container", children=[
                dash_svg.Svg(viewBox=f"0 0 {viewbox_size} {viewbox_size}", className="aqi-svg-gauge", children=[
                    dash_svg.Path(d=background_arc_path, className="aqi-gauge-track", style={'strokeWidth': stroke_width}),
                    dash_svg.Path(d=foreground_arc_path, className="aqi-gauge-value", 
                                  style={'stroke': aqi_color, 'strokeWidth': stroke_width}),
                    # Adjust y positions slightly for better centering with larger text/gauge
                    dash_svg.Text(f"{aqi_value}", x="50%", y="44%", dy=".1em", className="aqi-gauge-value-text"), 
                    dash_svg.Text(aqi_level, x="50%", y="64%", dy=".1em", className="aqi-gauge-level-text")  
                ])
            ]),
            html.P(f"Last Updated: {formatted_time}", className="aqi-obs-time-gauge")
        ])

    except APIError as e:
        print(f"APIError fetching current AQI for {query_city_for_api}: {e}") # For dev
        return html.Div(className="current-aqi-error-container", children=[
            html.P(f"Service error retrieving AQI for {selected_city}.", className="aqi-error-message")
        ])
    except Exception as e:
        print(f"General error updating current AQI for {query_city_for_api}: {e}") # For dev
        traceback.print_exc()
        return html.Div(className="current-aqi-error-container", children=[
            html.P(f"Error loading AQI data for {selected_city}.", className="aqi-error-message")
        ])
    
# --- Section 4: AQI Forecast Table ---

@app.callback(
    Output('aqi-forecast-table-content', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_aqi_forecast_table(selected_city):
    if not selected_city:
        return html.P("Select a city to view AQI forecast.", style={'textAlign': 'center', 'marginTop': '20px'})

    # predictor.py's generate_forecast expects the simple city name (e.g., "Delhi")
    # for model file lookup. selected_city from dropdown is already this simple name.
    
    try:
        # Default to 3 days forecast with residual correction.
        # Your predictor.py handles fetching weather for regressors.
        forecast_df = generate_forecast(
            target_city=selected_city, 
            days_ahead=3,  # Or use DEFAULT_FORECAST_DAYS from your predictor's config if accessible
            apply_residual_correction=True 
        )

        if forecast_df is None or forecast_df.empty:
            # This can happen if model loading fails, weather fails, or predictor returns None/empty
            return html.P(f"AQI forecast data is currently unavailable for {selected_city}.", 
                          className="forecast-error-message")

        formatted_forecast_list = format_forecast_for_ui(forecast_df)

        if not formatted_forecast_list:
            return html.P(f"Could not format forecast data for {selected_city}.",
                          className="forecast-error-message")

        table_header = [
            html.Thead(html.Tr([html.Th("Date"), html.Th("Predicted AQI")]))
        ]
        table_rows = [
            html.Tr([
                html.Td(item['date']), 
                html.Td(item['predicted_aqi'])
            ]) for item in formatted_forecast_list
        ]
        table_body = [html.Tbody(table_rows)]
        
        return html.Table(table_header + table_body, className="aqi-forecast-table")

    except ModelFileNotFoundError:
        # log.warning(f"ModelFileNotFoundError for {selected_city} forecast in Dash app.") # Example logging
        print(f"Dash App: ModelFileNotFoundError for {selected_city} forecast.") # Dev console
        return html.P(f"AQI forecast model not available for {selected_city}.", 
                      className="forecast-error-message")
    except APIError as e: # Catches API errors from weather_client called by predictor
        # log.error(f"APIError during forecast generation for {selected_city} in Dash app: {e}")
        print(f"Dash App: APIError during forecast for {selected_city}: {e}") # Dev console
        return html.P(f"Weather data for forecast unavailable for {selected_city}. Please try again.",
                      className="forecast-error-message")
    except PredictionError as pe: # Custom error from predictor.py
        # log.error(f"PredictionError for {selected_city} in Dash app: {pe}")
        print(f"Dash App: PredictionError for {selected_city}: {pe}") # Dev console
        return html.P(f"Could not generate forecast for {selected_city}: {pe}",
                      className="forecast-error-message")
    except Exception as e:
        # log.exception(f"General error generating forecast for {selected_city} in Dash app")
        print(f"Dash App: General error in forecast for {selected_city}: {e}") # Dev console
        traceback.print_exc() # Dev console for full traceback
        return html.P(f"Error generating AQI forecast for {selected_city}.",
                      className="forecast-error-message")

# --- Section 5: Current Pollutant Risk ---

@app.callback(
    Output('current-pollutant-risks-content', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_pollutant_risks_display(selected_city):
    if not selected_city:
        return html.P("Select a city to view current pollutant risks.", 
                      style={'textAlign': 'center', 'marginTop': '20px'})

    # get_current_pollutant_risks_for_city expects "City, India" format as per client.py structure
    query_city_for_api = f"{selected_city}, India" 

    try:
        risks_data = get_current_pollutant_risks_for_city(query_city_for_api)

        if not risks_data or 'error' in risks_data or not risks_data.get('risks'):
            error_message = "Pollutant risk data unavailable."
            if isinstance(risks_data, dict) and 'error' in risks_data:
                error_message = risks_data['error']
            elif not isinstance(risks_data, dict) or not risks_data.get('risks'): # No risks or malformed
                error_message = f"No specific pollutant risks identified or data is incomplete for {selected_city}."

            return html.Div([
                html.P(error_message, className="pollutant-risk-error")
            ], style={'textAlign': 'center', 'paddingTop': '30px'})

        # If we have risks
        risk_items = []
        if risks_data['risks']: # Ensure 'risks' key exists and is not empty
            for risk_statement in risks_data['risks']:
                # Attempt to highlight the pollutant part of the risk statement
                parts = risk_statement.split(":", 1)
                if len(parts) == 2:
                    pollutant_part = html.Strong(f"{parts[0]}:")
                    message_part = html.Span(parts[1])
                    risk_items.append(html.Li([pollutant_part, message_part], className="pollutant-risk-item"))
                else:
                    risk_items.append(html.Li(risk_statement, className="pollutant-risk-item"))
        else: # Should be caught by the check above, but as a fallback
            risk_items.append(html.Li("No significant pollutant risks identified at this time.", className="pollutant-risk-item-none"))
        
        # Optional: Display raw pollutant data (collapsible for tidiness)
        raw_pollutants = risks_data.get('pollutants', {})
        pollutant_details_children = []
        if raw_pollutants:
            for pol, data_val in raw_pollutants.items():
                if isinstance(data_val, dict) and 'v' in data_val: # Standard iaqi format
                     pollutant_details_children.append(html.P(f"{pol.upper()}: {data_val['v']}", className="raw-pollutant-value"))

        collapsible_content = []
        if pollutant_details_children:
            collapsible_content = [
                html.Details([
                    html.Summary("View Raw Pollutant Values", className="raw-pollutants-summary"),
                    html.Div(pollutant_details_children, className="raw-pollutants-container")
                ], className="pollutant-details-collapsible", open=False) # Collapsed by default
            ]

        return html.Div([
            html.H5("Key Health Advisories:", className="pollutant-risk-title"),
            html.Ul(risk_items, className="pollutant-risk-list")
        ] + collapsible_content) # Add collapsible section if it has content

    except APIError as e: # From the underlying get_city_aqi_data call
        print(f"APIError fetching pollutant risks for {query_city_for_api}: {e}")
        return html.P(f"Service error retrieving pollutant data for {selected_city}.", className="pollutant-risk-error")
    except Exception as e:
        print(f"General error updating pollutant risks for {query_city_for_api}: {e}")
        traceback.print_exc()
        return html.P(f"Error loading pollutant risk data for {selected_city}.", className="pollutant-risk-error")

# --- Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True)