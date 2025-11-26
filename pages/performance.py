# File: pages/performance.py

"""
This file contains the layout and callbacks for the live Performance Hub page.
"""
from dash import dcc, html, dash_table, Input, Output, callback
import dash
import pandas as pd
import psutil
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from shared_data import cpu_data, ram_data, net_data
import dash_bootstrap_components as dbc

# --- Imports and setup ---
try:
    from src.config_loader import read_last_n_log_lines
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PREDICTIONS_LOG_PATH = os.path.join(PROJECT_ROOT, "predictions_log.csv")
except ImportError:
    def read_last_n_log_lines(n=15): return ["Log reader not available."]
    PREDICTIONS_LOG_PATH = ""

dash.register_page(
    __name__,
    path='/performance',
    name='BreatheEasy Performance Hub',
    title='BreatheEasy | Performance',
    image='icon_performance.png'
)

APP_PROCESS = psutil.Process(os.getpid())

# --- Page Layout (Corrected) ---
layout = html.Div(className="performance-hub-shell", children=[
    html.Div(className="performance-header", children=[
        html.Img(
            id='hub-header-image',
            src=dash.get_asset_url('hub_header_dark.png'),
            className='hub-header-image',
            alt='BreatheEasy Live Performance Hub Logo'
        ),
        html.P("Real-time system metrics and application status dashboard."),
        dbc.Button(
            "Go back to main dashboard",
            href="/",
            className="mt-3",
            style={
                'backgroundColor': '#3374b1',
                'borderColor': '#3374b1',
                'color': '#F0F6FC'
            }
        )
    ]),
    html.Div(className="performance-grid", children=[
        html.Div(className="performance-column", children=[
            html.Div(className="widget-card-perf", children=[
                html.H4("Application Process Details"),
                html.Div(id='process-details-table-perf')
            ]),
            html.Div(className="widget-card-perf", children=[
                html.H4("Latest Forecasts Log"),
                html.Div(id='prediction-log-table-container-perf')
            ]),
        ]),
        html.Div(className="performance-column", children=[
            html.Div(className="widget-card-perf", children=[
                html.H4("Live System Health"),
                html.Div(className="scorecard-grid", children=[
                    html.Div(className="scorecard", children=[html.H5("CPU USAGE"), html.P(id='cpu-usage-text-perf')]),
                    html.Div(className="scorecard", children=[html.H5("MEMORY USAGE"), html.P(id='ram-usage-text-perf')]),
                    html.Div(className="scorecard", children=[html.H5("NETWORK IN"), html.P(id='net-in-text-perf')]),
                    html.Div(className="scorecard", children=[html.H5("NETWORK OUT"), html.P(id='net-out-text-perf')]),
                    html.Div(className="scorecard", children=[html.H5("UPTIME"), html.P(id='uptime-text-perf')]),
                ])
            ]),
            html.Div(className="widget-card-perf", children=[
                html.H4("System Performance Over Time"),
                dcc.Tabs(id="perf-graph-tabs", value='tab-cpu', className="performance-tabs", children=[
                    dcc.Tab(label='CPU', value='tab-cpu', className="performance-tab", selected_className="performance-tab--selected"),
                    dcc.Tab(label='Memory', value='tab-ram', className="performance-tab", selected_className="performance-tab--selected"),
                    dcc.Tab(label='Network', value='tab-net', className="performance-tab", selected_className="performance-tab--selected"),
                ]),
                html.Div(className="graph-container-perf", children=[
                    dcc.Graph(
                        id='performance-time-series-graph',
                        config={'displayModeBar': False},
                    ),
                    html.Div(id='graph-readout-container', className="graph-readout-container")
                ])
            ]),
            html.Div(className="widget-card-perf", children=[
                html.H4("Application Log (Last 15 lines)"),
                html.Details([
                    html.Summary("Click to view full application log", className="log-summary"),
                    html.Pre(id='app-log-container-perf', className='log-box')
                ])
            ]),
        ]),
    ]),
    dcc.Interval(id='performance-interval-timer-perf', interval=500, n_intervals=0)
])

# --- Helper function ---
def create_time_series_figure(x, y, name, color, y_axis_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', name=name, line=dict(color=color, width=2), fill='tozeroy', fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)'))
    fig.update_layout(margin=dict(l=40, r=20, t=10, b=20), plot_bgcolor='#161B22', paper_bgcolor='#161B22', font_color='#E6EDF3', xaxis=dict(showgrid=False), yaxis=dict(title=y_axis_title, gridcolor='rgba(255, 255, 255, 0.1)'), showlegend=False)
    return fig

# --- Main Callback ---
@callback(
    Output('performance-time-series-graph', 'figure'),
    Output('graph-readout-container', 'children'),
    Output('cpu-usage-text-perf', 'children'),
    Output('ram-usage-text-perf', 'children'),
    Output('net-in-text-perf', 'children'),
    Output('net-out-text-perf', 'children'),
    Output('uptime-text-perf', 'children'),
    Output('prediction-log-table-container-perf', 'children'),
    Output('app-log-container-perf', 'children'),
    Output('process-details-table-perf', 'children'),
    Input('performance-interval-timer-perf', 'n_intervals'),
    Input('perf-graph-tabs', 'value')
)
def update_live_metrics(n_intervals, active_tab):
    # --- 1. Get Latest Values from Global Data ---
    now = datetime.now()
    latest_cpu = cpu_data[-1][1] if cpu_data else 0
    latest_ram = ram_data[-1][1] if ram_data else 0
    latest_net_in = net_data[-1][1] if net_data else 0
    latest_net_out = net_data[-1][2] if net_data else 0

    cpu_output = html.Span([f"{latest_cpu:.1f}", html.Span(" %", className="scorecard-unit")])
    ram_output = html.Span([f"{latest_ram:.1f}", html.Span(" %", className="scorecard-unit")])
    net_in_output = html.Span([f"{latest_net_in:.1f}", html.Span(" KB/s", className="scorecard-unit")])
    net_out_output = html.Span([f"{latest_net_out:.1f}", html.Span(" KB/s", className="scorecard-unit")])

    table_style_header = {'backgroundColor': '#0D1117', 'color': '#E6EDF3', 'fontWeight': 'bold', 'borderBottom': '1px solid #58A6FF'}
    table_style_cell = {'backgroundColor': '#161B22', 'color': '#C9D1D9', 'border': 'none', 'textAlign': 'left', 'padding': '12px', 'whiteSpace': 'normal', 'height': 'auto', 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 200}
    try:
        uptime_delta = timedelta(seconds=int(now.timestamp() - APP_PROCESS.create_time()))
        uptime_output = str(uptime_delta)
        process_details = [{"Metric": k, "Value": v} for k, v in {"Process ID (PID)": str(APP_PROCESS.pid), "Status": APP_PROCESS.status(), "CPU % (Process)": f"{APP_PROCESS.cpu_percent() / psutil.cpu_count():.1f} %", "Memory Usage": f"{APP_PROCESS.memory_info().rss / (1024*1024):.1f} MB", "Uptime": uptime_output}.items()]
        process_table = dash_table.DataTable(columns=[{"name": i, "id": i} for i in process_details[0]], data=process_details, style_as_list_view=True, style_header={'display': 'none'}, style_cell=table_style_cell, style_data={'borderBottom': '1px solid rgba(255, 255, 255, 0.05)'}, style_cell_conditional=[{'if': {'column_id': 'Value'}, 'fontWeight': 'bold'}])
    except Exception as e:
        process_table = html.P(f"Could not read process info: {e}")
        uptime_output = "N/A"
    try:
        if os.path.exists(PREDICTIONS_LOG_PATH):
            log_df = pd.read_csv(PREDICTIONS_LOG_PATH).tail(5)
            log_table = dash_table.DataTable(columns=[{"name": col, "id": col} for col in log_df.columns], data=log_df.to_dict('records'), style_table={'overflowX': 'auto'}, style_header=table_style_header, style_cell=table_style_cell, style_data={'borderBottom': '1px solid rgba(255, 255, 255, 0.05)'})
        else:
            log_table = html.P("No predictions have been logged yet.")
    except Exception as e:
        log_table = html.P(f"Error reading prediction log: {e}")
    log_content = "".join(read_last_n_log_lines(15))
    
    figure_to_show = go.Figure()
    readout_to_show = []
    if active_tab == 'tab-cpu':
        x_vals, y_vals = zip(*cpu_data) if cpu_data else ([], [])
        figure_to_show = create_time_series_figure(x_vals, y_vals, 'CPU', '#58A6FF', 'Usage (%)')
        readout = html.Div([html.Span("Current: ", className="readout-label"), html.Span(f"{latest_cpu:.1f} %", className="readout-value")], className="readout-item")
        readout_to_show = [readout]
    elif active_tab == 'tab-ram':
        x_vals, y_vals = zip(*ram_data) if ram_data else ([], [])
        figure_to_show = create_time_series_figure(x_vals, y_vals, 'Memory', '#3FB950', 'Usage (%)')
        readout = html.Div([html.Span("Current: ", className="readout-label"), html.Span(f"{latest_ram:.1f} %", className="readout-value")], className="readout-item")
        readout_to_show = [readout]
    elif active_tab == 'tab-net':
        x_vals = [item[0] for item in net_data]
        y_in = [item[1] for item in net_data]
        y_out = [item[2] for item in net_data]
        y_out_negative = [-val for val in y_out]
        figure_to_show = go.Figure()
        figure_to_show.add_trace(go.Scatter(x=x_vals, y=y_out_negative, name='Sent', fill='tozeroy', line_color='#F778BA', hoverinfo='y'))
        figure_to_show.add_trace(go.Scatter(x=x_vals, y=y_in, name='Received', fill='tozeroy', line_color='#3FB950', hoverinfo='y'))
        figure_to_show.update_layout(margin=dict(l=40, r=20, t=10, b=20), plot_bgcolor='#161B22', paper_bgcolor='#161B22', font_color='#E6EDF3', xaxis=dict(showgrid=False), yaxis=dict(title='Rate (KB/s)', gridcolor='rgba(255, 255, 255, 0.1)', zerolinecolor='rgba(255,255,255,0.2)'), showlegend=False)
        readout_in = html.Div([html.Span([html.Span(className="readout-color-box", style={'backgroundColor': '#3FB950'}), "Received: "], className="readout-label"), html.Span(f"{latest_net_in:.1f} KB/s", className="readout-value")], className="readout-item")
        readout_out = html.Div([html.Span([html.Span(className="readout-color-box", style={'backgroundColor': '#F778BA'}), "Sent: "], className="readout-label"), html.Span(f"{latest_net_out:.1f} KB/s", className="readout-value")], className="readout-item")
        readout_to_show = [readout_in, readout_out]

    return (figure_to_show, readout_to_show, cpu_output, ram_output, net_in_output, 
            net_out_output, uptime_output, log_table, log_content, process_table)