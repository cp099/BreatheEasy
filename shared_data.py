# File: shared_data.py
"""
This file holds the data stores that are shared across the application
to avoid circular imports.
"""
import collections
import psutil
from datetime import datetime

# The deque data stores now live in this neutral file.
MAX_GRAPH_POINTS = 7200
cpu_data = collections.deque(maxlen=MAX_GRAPH_POINTS)
ram_data = collections.deque(maxlen=MAX_GRAPH_POINTS)
net_data = collections.deque(maxlen=MAX_GRAPH_POINTS)

# The helper variables also live here now.
last_net_io = psutil.net_io_counters()
last_net_time = datetime.now()