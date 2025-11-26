# File: pages/dashboard.py

"""
BreatheEasy - Main Dashboard

This file defines the user interface (UI) and server-side logic for the
BreatheEasy dashboard. It creates the layout of the web application, including
all widgets, and defines the callbacks that update the components in response
to user interactions (e.g., selecting a new city from the dropdown).

The application is structured as follows:
- Imports: Loads all necessary libraries and backend functions.
- App Initialization: Sets up the Dash app instance.
- App Layout: Defines the static HTML and Dash component structure.
- Callbacks: Contains the functions that make the dashboard dynamic and interactive.
"""

# --- Core Libraries ---
import dash
from dash import dcc, html, callback, Input, Output, State 
from dash.dependencies import Input, Output, State
import os
from dotenv import load_dotenv
import sys
import pandas as pd
import plotly.graph_objects as go
import traceback
import numpy as np
import math
import dash_svg

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --- Import Backend Functions & Project Modules ---
try:
    from src.api_integration.weather_client import get_current_weather
    from src.analysis.historical import get_historical_data_for_city
    from src.health_rules.info import AQI_DEFINITION, AQI_SCALE
    from src.exceptions import APIError
    from src.api_integration.client import get_current_aqi_for_city
    from src.health_rules.info import get_aqi_info 
    from src.api_integration.client import get_current_pollutant_risks_for_city
    from src.modeling.predictor import get_daily_summary_forecast
    from src.exceptions import APIError, ModelFileNotFoundError
except ImportError as e:
    print(f"CRITICAL ERROR importing backend modules: {e}")
    print("Ensure 'src' directory and all its submodules with __init__.py files are present and correct.")
    traceback.print_exc()

    def get_current_weather(city_name):
        print(f"Using DUMMY get_current_weather for {city_name}")
        return {"temp_c": "N/A", "condition_icon": None, "condition_text": "N/A", 
                "humidity": "N/A", "wind_kph": "N/A", "wind_dir": "", 
                "feelslike_c": "N/A", "pressure_mb": "N/A", "uv_index": "N/A",
                "error_message": "Backend weather client not loaded"} 

    def get_historical_data_for_city(city_name):
        print(f"Using DUMMY get_historical_data_for_city for {city_name}")
        return pd.DataFrame() 
    
    def get_current_aqi_for_city(city_name):
        print(f"Using DUMMY get_current_aqi_for_city for {city_name}")
        if city_name == "Delhi, India": 
            return {'city': 'Delhi', 'aqi': 55, 'station': 'Dummy Station, Delhi', 'time': '2023-10-27 10:00:00'}
        elif city_name == "Mumbai, India":
            return {'city': 'Mumbai', 'aqi': 155, 'station': 'Dummy Station, Mumbai', 'time': '2023-10-27 10:05:00'}
        else:
            return {'city': city_name.split(',')[0], 'aqi': None, 'station': 'Unknown station', 'time': None, 'error': 'Station not found or API error'}

    def get_aqi_info(aqi_value):
        print(f"Using DUMMY get_aqi_info for AQI: {aqi_value}")
        if aqi_value is None: return {'level': 'N/A', 'range': '-', 'color': '#DDDDDD', 'implications': 'AQI value not available.'}
        if aqi_value <= 50: return {'level': 'Good', 'range': '0-50', 'color': '#A8E05F', 'implications': 'Minimal impact.'} 
        if aqi_value <= 100: return {'level': 'Satisfactory', 'range': '51-100', 'color': '#D4E46A', 'implications': 'Minor breathing discomfort.'} 
        if aqi_value <= 200: return {'level': 'Moderate', 'range': '101-200', 'color': '#FDD74B', 'implications': 'Breathing discomfort.'} 
        return {'level': 'Poor', 'range': '201+', 'color': '#FFA500', 'implications': 'Significant breathing discomfort.'} 
    
    def get_current_pollutant_risks_for_city(city_name):
        print(f"Using DUMMY get_current_pollutant_risks_for_city for {city_name}")
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
                'risks': ["No significant pollutant risks identified at this time."] 
            }
        else: 
            return {'city': city_simple, 'time': None, 'pollutants': {}, 'risks': [], 'error': 'Pollutant data unavailable for this city.'}
        
    def get_predicted_weekly_risks(city_name, days_ahead=3):
        print(f"Using DUMMY get_predicted_weekly_risks for {city_name}")
        dummy_risks = []
        base_date = pd.Timestamp.now().normalize()
        aqi_levels_dummy = [
            (200, {'level': 'Moderate', 'color': '#FDD74B', 'implications': 'Dummy moderate implications.'}),
            (239, {'level': 'Poor', 'color': '#FFA500', 'implications': 'Dummy poor implications.'}),
            (235, {'level': 'Poor', 'color': '#FFA500', 'implications': 'Dummy poor implications.'})
        ]
        for i in range(days_ahead):
            aqi_val, cat_info = aqi_levels_dummy[i % len(aqi_levels_dummy)] 
            dummy_risks.append({
                'date': (base_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'predicted_aqi': aqi_val + i*2, 
                'level': cat_info['level'],
                'color': cat_info['color'],
                'implications': cat_info['implications']
            })
        return dummy_risks

    class ModelFileNotFoundError(Exception): pass

    AQI_DEFINITION = "AQI definition not loaded (dummy). Check imports."
    AQI_SCALE = [{'level': 'Error', 'range': 'N/A', 'color': '#CCCCCC', 
                  'implications': 'AQI Scale not loaded (dummy). Check imports.'}]
    
    class APIError(Exception): pass 


# --- Application Configuration ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    print("Warning: .env file not found. API calls might fail or use defaults.")

TARGET_CITIES = ['Bangalore', 'Chennai', 'Kolkata', 'Mumbai']

# --- Initialize the Dash App ---
dash.register_page(
    __name__,
    path='/',
    name='BreatheEasy Dashboard',
    title='BreatheEasy | Dashboard',
    image='icon_dashboard.png'
)

# --- App Layout Definition ---
layout = html.Div(id='app-container', className="app-shell", children=[
    dcc.Store(id='theme-store', storage_type='local', data='light'),
    html.Button("🌙", id='theme-toggle-button', className='theme-toggle'),
    html.Div(className="content-above-footer", children=[
        # 1. Page Header with Logo
        html.Div(className="page-header", children=[
            html.Img(id='logo-image', # ADD THIS ID
                src=dash.get_asset_url('logo_light.png'), 
                className="logo-image", 
                alt="BreatheEasy Project Logo")
        ]),

        # 2. Main Control Bar with Dropdown and Weather
        html.Div(className="control-bar", children=[
            html.Div(className="city-dropdown-container", children=[
                dcc.Dropdown(
                    id='city-dropdown',
                    options=[{'label': city, 'value': city} for city in TARGET_CITIES],
                    value=TARGET_CITIES[0],
                    placeholder="Select a city",
                    clearable=False,
                    className="city-dropdown"
                )
            ]),
            html.Div(id='current-weather-display', className="current-weather-display")
        ]),

        # 3. Main Content Grid for all widgets
        html.Div(className="main-content-grid", children=[
            # --- Row 1 ---
            html.Div(className="widget-card", id="section-1-hist-summary", children=[
                html.H3("Historical Summary"),
                dcc.Graph(
                    id='historical-aqi-trend-graph',
                    figure={},
                    config={'responsive': True, 'displayModeBar': False},
                    className='flex-graph-container'
                )
            ]),
            html.Div(className="widget-card", id="section-3-curr-aqi", children=[
                html.H3("Current AQI"),
                html.Div(id='current-aqi-details-content', className='current-aqi-widget-content')
            ]),
            html.Div(className="widget-card", id="section-5-pollutant-risks", children=[
                html.H3("Current Pollutant Risks"),
                html.Div(id='current-pollutant-risks-content', className='pollutant-risks-widget-content') 
            ]),

            # --- Row 2 ---
            html.Div(className="widget-card", id="section-2-edu-info", children=[
                html.H3("AQI Educational Info"),
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
                html.H3("AQI Forecast (Next 3 Days)"),
                html.Div(id='aqi-forecast-table-content', className='forecast-widget-content')
            ]),
            html.Div(className="widget-card", id="section-6-weekly-risks", children=[
                html.H3("Predicted Weekly Risks & Advisories"),
                html.Div(id='predicted-weekly-risks-content', className='predicted-risks-widget-content'),
            ]) 
        ]) 
    ]), 

    # 4. Page Footer
    html.Div(className="page-footer", children=[
        html.P("Project Team: Arnav Vaidya, Chirag P Patil, Kimaya Anand | School: Delhi Public School Bangalore South"),
        html.P("Copyright © 2025 BreatheEasy Project Team. Licensed under MIT License.")
    ]) 
])

# --- Callbacks ---
@callback(
    Output('current-weather-display', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_current_weather(selected_city):
    """Fetches and displays the current weather for the selected city."""
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
        print(f"Handled APIError for {selected_city}: {e}")     
        return get_default_weather_layout(city_name_text=selected_city, error_message="Weather service unavailable.")
    except Exception as e:
        print(f"General error fetching weather for {selected_city}: {e}")
        traceback.print_exc()
        return get_default_weather_layout(city_name_text=selected_city, error_message="Error loading weather.")

@callback(
    Output('historical-aqi-trend-graph', 'figure'),
    Input('city-dropdown', 'value'),
    Input('theme-store', 'data') 
)
def update_historical_trend_graph(selected_city, current_theme):
    """
    Generates the historical AQI trend graph from the final daily feature dataset,
    with robust data processing.
    """
    def create_placeholder_figure(message_text):
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text=message_text, showarrow=False, font=dict(size=14, color="#0A4D68"))],
            xaxis_visible=False, yaxis_visible=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    if not selected_city:
        return create_placeholder_figure("Select a city to view historical AQI trend.")

    try:
        # Step 1: Get the data using our new function
        df_city_raw = get_historical_data_for_city(selected_city)
        if df_city_raw.empty:
            return create_placeholder_figure(f"No historical data available for {selected_city}.")

        # Step 2: Resample the data to daily, similar to the new logic
        daily_aqi_series = df_city_raw['AQI'].resample('D').mean().dropna()

        # Step 3: Apply the ROBUST processing from your OLD working code
        df_trend = daily_aqi_series.reset_index()
        df_trend.columns = ['Date', 'AQI'] 

        df_trend['Date'] = pd.to_datetime(df_trend['Date'], errors='coerce')
        df_trend['AQI'] = pd.to_numeric(df_trend['AQI'], errors='coerce')
        df_trend.dropna(subset=['Date', 'AQI'], inplace=True)
        
        if df_trend.empty:
            return create_placeholder_figure(f"No valid data to plot for {selected_city}.")

        x_values_for_plot = df_trend['Date'].tolist()
        y_values_for_plot = df_trend['AQI'].tolist()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=x_values_for_plot, y=y_values_for_plot, mode='lines', name='Daily Mean AQI',
                line=dict(color='var(--accent-primary)') 
            )
        ])

        graph_template = 'plotly_dark' if current_theme == 'dark' else 'plotly_white'
        
        fig.update_layout(
            title_text=f"Historical Daily Mean AQI for {selected_city}", title_x=0.5,
            xaxis_title="Date", yaxis_title="AQI Value",
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor='rgba(0,0,0,0)',     
            paper_bgcolor='rgba(0,0,0,0)',    
            template=graph_template         
        )
        return fig
        
    except Exception as e:
        print(f"ERROR IN HISTORICAL TREND CALLBACK for {selected_city}: {e}")
        traceback.print_exc()
        return create_placeholder_figure(f"Error displaying trend for {selected_city}.")

# --- SVG Gauge Helper Function ---
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

@callback(
    Output('current-aqi-details-content', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_current_aqi_details(selected_city): 
    """Fetches the current AQI and renders the SVG gauge for the selected city."""
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

        max_aqi_on_scale = 500.0
        current_aqi_clamped = max(0, min(float(aqi_value), max_aqi_on_scale)) 
        percentage = current_aqi_clamped / max_aqi_on_scale
        
        gauge_start_angle_deg = -225 
        gauge_total_sweep_deg = 270 
        value_end_angle_deg = gauge_start_angle_deg + (percentage * gauge_total_sweep_deg)

        viewbox_size = 280 
        center_xy = viewbox_size / 2
        radius = 115   
        stroke_width = 22 

        background_arc_path = describe_arc(center_xy, center_xy, radius, gauge_start_angle_deg, gauge_start_angle_deg + gauge_total_sweep_deg)
        foreground_arc_path = describe_arc(center_xy, center_xy, radius, gauge_start_angle_deg, value_end_angle_deg)
        
        return html.Div(className="aqi-gauge-wrapper", children=[
            html.H4(selected_city, className="aqi-city-name-highlight"),
            html.Div(className="aqi-gauge-svg-container", children=[
                dash_svg.Svg(viewBox=f"0 0 {viewbox_size} {viewbox_size}", className="aqi-svg-gauge", children=[
                    dash_svg.Path(d=background_arc_path, className="aqi-gauge-track", style={'strokeWidth': stroke_width}),
                    dash_svg.Path(d=foreground_arc_path, className="aqi-gauge-value", 
                                  style={'stroke': aqi_color, 'strokeWidth': stroke_width}),
                    dash_svg.Text(f"{aqi_value}", x="50%", y="44%", dy=".1em", className="aqi-gauge-value-text"), 
                    dash_svg.Text(aqi_level, x="50%", y="64%", dy=".1em", className="aqi-gauge-level-text")  
                ])
            ]),
            html.P(f"Last Updated: {formatted_time}", className="aqi-obs-time-gauge")
        ])

    except APIError as e:
        print(f"APIError fetching current AQI for {query_city_for_api}: {e}") 
        return html.Div(className="current-aqi-error-container", children=[
            html.P(f"Service error retrieving AQI for {selected_city}.", className="aqi-error-message")
        ])
    except Exception as e:
        print(f"General error updating current AQI for {query_city_for_api}: {e}") 
        traceback.print_exc()
        return html.Div(className="current-aqi-error-container", children=[
            html.P(f"Error loading AQI data for {selected_city}.", className="aqi-error-message")
        ])

# --- Section 4 & 6: Unified AQI Forecast and Predicted Risks ---
@callback(
    Output('aqi-forecast-table-content', 'children'),
    Output('predicted-weekly-risks-content', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_all_forecast_widgets(selected_city):
    """
    A single, unified callback to generate and display the calibrated forecast
    in both Section 4 and Section 6, ensuring consistency.
    """
    if not selected_city:
        placeholder = html.P("Select a city to view forecast.", style={'textAlign': 'center', 'marginTop': '20px'})
        return placeholder, placeholder

    try:
        risks_and_forecast_list = get_daily_summary_forecast(selected_city, days_ahead=3)

        if not risks_and_forecast_list:
            error_message = html.P(f"Forecast data is currently unavailable for {selected_city}.", className="forecast-error-message")
            return error_message, error_message

        forecast_cards_sec4 = []
        risk_cards_sec6 = []
        
        for day_data in risks_and_forecast_list:
            aqi_val = day_data.get('predicted_aqi')
            level = day_data.get('level', 'N/A')
            color = day_data.get('color', '#DDDDDD')
            date_str = day_data.get('date', 'N/A')
            implications = day_data.get('implications', 'No specific implications provided.')
            
            # Create Card for Section 4 (Simple Forecast)
            card_style_sec4 = {'borderLeft': f"7px solid {color}", 'padding': '12px 15px', 'borderRadius': '6px', 'backgroundColor': f"{color}1A"}
            forecast_cards_sec4.append(
                html.Div(style=card_style_sec4, className="forecast-day-card", children=[
                    html.Div(className="forecast-card-header", children=[
                        html.Strong(date_str, className="forecast-date"),
                        html.Span(children=["AQI: ", html.Span(f"{aqi_val}", style={'fontWeight': 'bold'}), f" ({level})"], className="forecast-aqi-level", style={'color': color})
                    ])
                ])
            )
            
            # Create Card for Section 6 (Detailed Risks)
            card_style_sec6 = {'borderLeft': f"7px solid {color}", 'padding': '10px 15px', 'borderRadius': '6px', 'backgroundColor': f"{color}1A"}
            risk_cards_sec6.append(
                html.Div(style=card_style_sec6, className="predicted-risk-day-card", children=[
                    html.Div(className="predicted-risk-header", children=[
                        html.Strong(date_str, className="predicted-risk-date"),
                        html.Span(f"AQI: {aqi_val} ({level})", className="predicted-risk-aqi-level", style={'color': color})
                    ]),
                    html.P(implications, className="predicted-risk-implications")
                ])
            )

        return forecast_cards_sec4, risk_cards_sec6

    except Exception as e:
        print(f"Error in unified forecast callback for {selected_city}: {e}")
        traceback.print_exc()
        error_message = html.P("An error occurred while generating the forecast.", className="forecast-error-message")
        return error_message, error_message

@callback(
    Output('current-pollutant-risks-content', 'children'),
    [Input('city-dropdown', 'value')]
)
def update_pollutant_risks_display(selected_city):
    """Fetches current pollutant data and displays interpreted health risks."""
    if not selected_city:
        return html.P("Select a city to view current pollutant risks.", 
                      style={'textAlign': 'center', 'marginTop': '20px'})

    query_city_for_api = f"{selected_city}, India" 

    try:
        risks_data = get_current_pollutant_risks_for_city(query_city_for_api)

        if not risks_data or 'error' in risks_data or not risks_data.get('risks'):
            error_message = "Pollutant risk data unavailable."
            if isinstance(risks_data, dict) and 'error' in risks_data:
                error_message = risks_data['error']
            elif not isinstance(risks_data, dict) or not risks_data.get('risks'): 
                error_message = f"No specific pollutant risks identified or data is incomplete for {selected_city}."

            return html.Div([
                html.P(error_message, className="pollutant-risk-error")
            ], style={'textAlign': 'center', 'paddingTop': '30px'})

        risk_items = []
        if risks_data['risks']: 
            for risk_statement in risks_data['risks']:
                parts = risk_statement.split(":", 1)
                if len(parts) == 2:
                    pollutant_part = html.Strong(f"{parts[0]}:")
                    message_part = html.Span(parts[1])
                    risk_items.append(html.Li([pollutant_part, message_part], className="pollutant-risk-item"))
                else:
                    risk_items.append(html.Li(risk_statement, className="pollutant-risk-item"))
        else: 
            risk_items.append(html.Li("No significant pollutant risks identified at this time.", className="pollutant-risk-item-none"))
        
        raw_pollutants = risks_data.get('pollutants', {})
        pollutant_details_children = []
        if raw_pollutants:
            for pol_code, data_dict in raw_pollutants.items():
                if isinstance(data_dict, dict) and 'v' in data_dict:
                    value = data_dict['v']
                    try:
                        if isinstance(value, float) and (abs(value - round(value, 2)) > 1e-9 or len(str(value).split('.')[-1]) > 2) : 
                            display_value = f"{value:.1f}"
                        elif isinstance(value, float):
                             display_value = f"{int(value)}" 
                        else:
                            display_value = str(value)
                    except:
                        display_value = str(value) 

                    pollutant_details_children.append(
                        html.Div(className="pollutant-pill", children=[
                            html.Span(f"{pol_code.upper()}: ", className="pollutant-pill-name"), 
                            html.Span(display_value, className="pollutant-pill-value")
                        ])
                    )

        collapsible_content = []
        if pollutant_details_children:
            collapsible_content = [ 
                html.Details([
                    html.Summary("View Raw Pollutant Values", className="raw-pollutants-summary"),
                    html.Div(pollutant_details_children, className="raw-pollutants-container")
                ], className="pollutant-details-collapsible", open=False) 
            ]

        return html.Div([
            html.H5("Key Health Advisories:", className="pollutant-risk-title"),
            html.Ul(risk_items, className="pollutant-risk-list")
        ] + collapsible_content) 

    except APIError as e: 
        print(f"APIError fetching pollutant risks for {query_city_for_api}: {e}")
        return html.P(f"Service error retrieving pollutant data for {selected_city}.", className="pollutant-risk-error")
    except Exception as e:
        print(f"General error updating pollutant risks for {query_city_for_api}: {e}")
        traceback.print_exc()
        return html.P(f"Error loading pollutant risk data for {selected_city}.", className="pollutant-risk-error")

# --- THEME TOGGLE CALLBACKS ---

@callback(
    Output('theme-store', 'data'),
    Input('theme-toggle-button', 'n_clicks'),
    State('theme-store', 'data'),
    prevent_initial_call=True
)
def toggle_theme_store(n_clicks, current_theme):
    """Toggles the theme between light and dark and saves it in the store."""
    if current_theme == 'light':
        return 'dark'
    else:
        return 'light'

@callback(
    Output('app-container', 'className'),
    Input('theme-store', 'data')
)
def update_app_theme_class(theme):
    """Applies the .dark-theme class to the main app container."""
    if theme == 'dark':
        return "app-shell dark-theme"
    return "app-shell"

@callback(
    Output('logo-image', 'src'),
    Input('theme-store', 'data')
)
def update_logo_src(theme):
    """Switches the logo image file based on the current theme."""
    if theme == 'dark':
        return dash.get_asset_url('logo_dark.png')
    else:
        return dash.get_asset_url('logo_light.png')
