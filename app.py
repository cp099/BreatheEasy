# File: app.py (CORRECTED)
"""
Main application file. Imports and updates the central data stores.
"""
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import os
import psutil
from datetime import datetime

# --- Import from the new shared data file ---
from shared_data import cpu_data, ram_data, net_data, last_net_io, last_net_time

# Initialize the Dash App
app = dash.Dash(
    __name__, 
    use_pages=True, 
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.MINTY], 
    assets_folder='assets'
)
server = app.server
app.title = "BreatheEasy"

# Main App Layout
app.layout = html.Div([
    dash.page_container,
    dcc.Interval(id='background-data-interval', interval=500),
    html.Div(id='dummy-output', style={'display': 'none'})
])

# Background data collection callback
@app.callback(
    Output('dummy-output', 'children'),
    Input('background-data-interval', 'n_intervals')
)
def update_background_data(n):
    # This now needs to be a mutable global, which is fine for deques
    global last_net_io, last_net_time
    
    now = datetime.now()
    # ... (rest of the logic is the same)
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    
    current_net_io = psutil.net_io_counters()
    time_delta = (now - last_net_time).total_seconds()
    if time_delta > 0:
        net_in_rate = (current_net_io.bytes_recv - last_net_io.bytes_recv) / time_delta / 1024
        net_out_rate = (current_net_io.bytes_sent - last_net_io.bytes_sent) / time_delta / 1024
    else:
        net_in_rate, net_out_rate = 0, 0
    
    last_net_io = current_net_io
    last_net_time = now
    
    cpu_data.append((now, cpu_percent))
    ram_data.append((now, ram_percent))
    net_data.append((now, net_in_rate, net_out_rate))
    
    return ""

# Run the Application
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=True)