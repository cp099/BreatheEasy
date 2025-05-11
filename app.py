import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State # State not used yet, but good to have
import os
from dotenv import load_dotenv
import sys
import pandas as pd 
import plotly.express as px 
from plotly import graph_objects as go
import traceback
import numpy as np

# --- Load environment variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print("Warning: .env file not found. API calls might fail.")

# --- Add src directory to Python path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # Only one needed if src is at root


# --- Import backend functions ---
try:
    from api_integration.weather_client import get_current_weather
    from analysis.historical import get_city_aqi_trend_data
    from exceptions import APIError
except ImportError as e:
    print(f"Error importing backend modules: {e}. Ensure 'src' is in PYTHONPATH or sys.path is correct.")
    def get_current_weather(city_name):
        print(f"Using DUMMY get_current_weather for {city_name}")
        return {
            "temp_c": "25", "condition": {"text": "Dummy Condition", "icon": "//cdn.weatherapi.com/weather/64x64/day/113.png"}, # Dummy icon
            "humidity": "60", "wind_kph": "10", "error": "Backend not loaded (dummy data)"
        }
    def get_city_aqi_trend_data(city_name):
        print(f"Using DUMMY get_city_aqi_trend_data for {city_name}")
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        aqi_values = [50, 55, 60, 52, 58]
        return pd.Series(data=aqi_values, index=dates, name="AQI")
    APIError = Exception

# --- Hardcoded City List for Dropdown ---
HARDCODED_CITIES = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad']

# --- Initialize Dash App ---
app = dash.Dash(__name__, assets_folder='assets')
app.title = "BreatheEasy"

# --- App Layout ---
app.layout = html.Div(className="app-shell", children=[
    html.Div(className="page-header", children=[
        # html.Img(src=app.get_asset_url('logo_placeholder.png'), className="logo-image", alt="BreatheEasy Logo")
        html.Div("LOGO", className="logo-placeholder") # Using Div placeholder as per screenshot
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
        html.Div(className="widget-card", id="section-1-hist-summary", children=[html.H3("Section 1: Historical Summary"), dcc.Graph(id='historical-aqi-trend-graph',
                figure={}, # Start with an empty figure
                config={'responsive': True, 'displayModeBar': False}, 
                style={'height': '100%', 'width': '100%'})
                ]), 
        html.Div(className="widget-card", id="section-3-curr-aqi", children=[html.H3("Section 3: Current AQI"), html.P("Content for Current AQI...")]),              # Item 2 (was Section 2)
        html.Div(className="widget-card", id="section-5-pollutant-risks", children=[html.H3("Section 5: Current Pollutant Risks"), html.P("Content for Pollutant Risks...")]), # Item 3 (was Section 3)

        # Row 2
        html.Div(className="widget-card", id="section-2-edu-info", children=[html.H3("Section 2: Educational Info"), html.P("Content for Edu. Info...")]),          # Item 4 (was Section 4)
        html.Div(className="widget-card", id="section-4-aqi-forecast", children=[html.H3("Section 4: AQI Forecast"), html.P("Content for AQI Forecast...")]),       # Item 5 (was Section 5)
        html.Div(className="widget-card", id="section-6-weekly-risks", children=[html.H3("Section 6: Predicted Weekly Risks"), html.P("Content for Weekly Risks...")]),     # Item 6
    ]),

    html.Div(className="page-footer", children=[
        html.P("Project Team: Arnav Vidya, Chirag P Patil, Kimaya Anand | School: Delhi Public School Bangalore South"),
        html.P("Copyright © 2025 BreatheEasy/Delhi Public School Bangalore South. Licensed under MIT License.")
    ])
])

# --- Callbacks ---
# Ensure html from dash is imported: from dash import html
# Ensure APIError from your exceptions is imported

@app.callback(
    Output('current-weather-display', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_current_weather(selected_city):
    # Helper function to generate placeholder/error weather display
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
            html.Div(className="weather-text-info-expanded", children=[ # Assuming you want to use weather-text-info-expanded consistently
                html.P(city_name_text, className="weather-city"),
                html.P("-°C", className="weather-temp"),
                html.P(condition_display_children, className="weather-condition", style=condition_style),
                html.Div(className="weather-details-row", children=[
                    html.P("Humidity: - %", className="weather-details"),
                    html.P("Wind: - kph", className="weather-details")
                    # Add other placeholder details if desired
                ])
            ])
        ]

    if not selected_city:
        return get_default_weather_layout()

    try:
        query_city_for_api = f"{selected_city}, India"
        weather_data = get_current_weather(query_city_for_api) # Backend call

        if weather_data and isinstance(weather_data, dict) and 'temp_c' in weather_data:
            icon_url_path = weather_data.get('condition_icon')
            condition_text = weather_data.get('condition_text', "Not available")

            icon_url = ""
            if icon_url_path:
                icon_url = "https:" + icon_url_path if icon_url_path.startswith("//") else icon_url_path
            
            if not condition_text or str(condition_text).strip() == "":
                condition_text = "Not available"

            return [
                html.Div(className="weather-icon-container", children=[
                    html.Img(
                        src=icon_url,
                        alt=condition_text if condition_text != "Not available" else "Weather icon",
                        className="weather-icon",
                        style={'display': 'block' if icon_url else 'none'}
                    )
                ]),
                html.Div(className="weather-text-info-expanded", children=[
                    html.P(f"{selected_city}", className="weather-city"),
                    html.P(f"{weather_data.get('temp_c', '-')}°C", className="weather-temp"),
                    html.P(f"{condition_text}", className="weather-condition"),
                    html.Div(className="weather-details-row", children=[
                        html.P(f"Humidity: {weather_data.get('humidity', '-')} %", className="weather-details"),
                        html.P(f"Wind: {weather_data.get('wind_kph', '-')} kph {weather_data.get('wind_dir', '')}", className="weather-details"),
                        html.P(f"Feels like: {weather_data.get('feelslike_c', '-')}°C", className="weather-details"),
                        html.P(f"Pressure: {weather_data.get('pressure_mb', '-')} mb", className="weather-details"),
                        html.P(f"UV Index: {weather_data.get('uv_index', '-')}", className="weather-details"),
                    ])
                ])
            ]
        else:
            # Handle cases where API returns data but not the expected structure, or an error within the data
            error_msg = "Weather data not available"
            if weather_data and isinstance(weather_data, dict):
                if weather_data.get("error_message"): 
                    error_msg = weather_data.get("error_message")
                elif weather_data.get("error"): # From WeatherAPI.com error response
                    error_detail = weather_data.get("error")
                    error_msg = error_detail.get("message", "Unknown API error") if isinstance(error_detail, dict) else str(error_detail)
            return get_default_weather_layout(city_name_text=selected_city, error_message=error_msg)

    except APIError as e: # Your custom API error
        # Log this error to your file using your logger if set up, e.g., log.error(...)
        print(f"Handled APIError for {selected_city}: {e}") # Keep a concise print for console during dev
        return get_default_weather_layout(city_name_text=selected_city, error_message="Service unavailable.")
    except Exception as e:
        # Log this error to your file, e.g., log.exception(...)
        print(f"General error fetching weather for {selected_city}: {e}") # Keep for dev
        # Consider importing 'traceback' and calling traceback.print_exc() here for dev
        return get_default_weather_layout(city_name_text=selected_city, error_message="Error loading weather.")
    
@app.callback(
    Output('historical-aqi-trend-graph', 'figure'),
    [Input('city-dropdown', 'value')]
)
def update_historical_trend_graph(selected_city):
    # Placeholder figure for "No City Selected" or "No Data"
    def create_placeholder_figure(message_text, height=300):
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text=message_text, showarrow=False, font=dict(size=14, color="#0A4D68"))],
            xaxis_visible=False, yaxis_visible=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=height
        )
        return fig

    if not selected_city:
        return create_placeholder_figure("Select a city to view historical AQI trend")

    try:
        aqi_trend_series = get_city_aqi_trend_data(selected_city)

        if aqi_trend_series is None or aqi_trend_series.empty:
            # Log this occurrence if you have a logger: log.warning(f"No historical data for {selected_city}")
            return create_placeholder_figure(f"No historical data available for {selected_city}")
        
        df_trend = aqi_trend_series.reset_index()
        if 'Date' not in df_trend.columns or 'AQI' not in df_trend.columns:
            if len(df_trend.columns) == 2:
                 df_trend.columns = ['Date', 'AQI']
            else:
                # Log this error: log.error(f"Unexpected DataFrame columns for {selected_city}: {df_trend.columns.tolist()}")
                raise ValueError(f"DataFrame columns not as expected ('Date', 'AQI').")

        # Data Cleaning and Validation
        df_trend['Date'] = pd.to_datetime(df_trend['Date'], errors='coerce')
        df_trend['AQI'] = pd.to_numeric(df_trend['AQI'], errors='coerce')
        
        if np.isinf(df_trend['AQI']).any():
            # Log this: log.warning(f"Infinity values found in AQI for {selected_city}. Replacing with NaN.")
            df_trend.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df_trend.dropna(subset=['Date', 'AQI'], inplace=True)
        df_trend = df_trend.sort_values(by='Date').reset_index(drop=True)
        
        if df_trend.empty:
            # Log this: log.warning(f"Data became empty after cleaning for {selected_city}.")
            return create_placeholder_figure(f"No valid data after cleaning for {selected_city}")

        actual_aqi_min = df_trend['AQI'].min()
        actual_aqi_max = df_trend['AQI'].max()

        x_values_for_plot = df_trend['Date'].tolist()
        y_values_for_plot = df_trend['AQI'].tolist()

        fig = go.Figure(data=[
            go.Scatter(
                x=x_values_for_plot,
                y=y_values_for_plot,
                mode='lines',
                name='AQI Trend', # This name appears if you have a legend enabled
                line_shape='linear' # Default for 'lines', but explicit
            )
        ])

        # Determine y-axis range with a buffer, ensuring min is not below 0
        yaxis_buffer = 0.1 * (actual_aqi_max - actual_aqi_min) if (actual_aqi_max - actual_aqi_min) > 0 else 10
        yaxis_min_range = max(0, actual_aqi_min - yaxis_buffer)
        yaxis_max_range = actual_aqi_max + yaxis_buffer
        
        # Consistent title for the graph (the H3 in the widget card serves as "Section X: Title")
        graph_title = f"AQI Trend for {selected_city}" 

        fig.update_layout(
            title_text=graph_title,
            title_x=0.5, # Center title
            title_font_size=16, # Slightly smaller if H3 is main title
            height=400,
            xaxis_title="Date",
            yaxis_title="AQI Value",
            yaxis_range=[yaxis_min_range, yaxis_max_range],
            margin=dict(l=60, r=30, t=60, b=60), # Left, Right, Top, Bottom margins
            plot_bgcolor='rgba(255,255,255,0.9)', # Slightly opaque white plot area
            paper_bgcolor='rgba(0,0,0,0)',       # Transparent paper to blend with widget
            font_color="#0A4D68"                 # Consistent font color
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightSteelBlue')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightSteelBlue')
        
        fig.update_traces(
            line=dict(color='#007bff', width=1.5), # A standard blue color
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>AQI</b>: %{y:.0f}<extra></extra>" # Format AQI as integer
        )
        
        return fig

    except Exception as e:
        # Use your logging system here, e.g., log.exception(f"Error generating historical trend graph for {selected_city}")
        print(f"ERROR IN HISTORICAL TREND CALLBACK for {selected_city}:") # Keep for dev console
        traceback.print_exc() # Keep for dev console to see traceback
        return create_placeholder_figure(f"Error displaying trend for {selected_city}", height=400)


# --- Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True)