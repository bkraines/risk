import sys
import os

# --- Start Fix ---
# Get the absolute path of the directory containing this script (dashboard)
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root directory (one level up)
project_root = os.path.dirname(dashboard_dir)

# Add the project root to the beginning of sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Fix ---

# Optional: Keep debug prints to verify
print("--- Debug Info ---")
print(f"Current Working Directory: {os.getcwd()}")
print("sys.path (after modification):") # Added label
for p in sys.path:
    print(f"  - {p}")
print("--- End Debug Info ---")

import streamlit as st

from risk_lib.config import HALFLIFES
from risk_lib.data import get_factor_data

from dashboard.feedback import build_streamlit_dashboard as market_feedback
from dashboard.timeseries import build_dashboard_vol as time_series
from dashboard.corr_mds_dashboard import build_streamlit_dashboard as corr_mds
from dashboard.factor_master_dashboard import build_streamlit_dashboard as factor_master
# from risk_lib.util import add_sidebar_defaults

factor_data = get_factor_data()

pg = st.navigation([st.Page(lambda: market_feedback(factor_data), 
                            title='Market Feedback',
                            url_path='feedback'), 
                    
                    st.Page(lambda: time_series(factor_data), 
                            title='Time Series',
                            url_path='timeseries'),
                    
                    st.Page(lambda: corr_mds(factor_data),
                            title='Correlation Projection',
                            url_path='correlation'),
                    
                    st.Page(lambda: factor_master(),
                            title='Factor Master',
                            url_path='factor_master'),
                    ])

# add_sidebar_defaults()

pg.run()

