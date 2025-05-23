# Define the portfolios using an OrderedDict
import pandas as pd
from collections import OrderedDict

# Prepare the fixed weights data with index set directly in DataFrame constructor
# fixed_weights_data = pd.DataFrame(
#     {
#         'MWTIX': [1, 0.5, 0.75],
#         'SHY': [0, 0.5, 0.1],
#     },
#     index=pd.DatetimeIndex(['2015-01-31', '2024-10-31', '2024-06-30'], name='date')
# )

fixed_weights_data = pd.read_excel('factor_master.xlsx', sheet_name='tracking').fillna(0).set_index('date')

# TODO: Assert x_tickers is subset of tickers
x_tickers = [ 'AGG', 'IEF', 'VMBS', 'IEI', 'LQD', 'TLT', 'TIP', 'SHY', 'EMB']
x_tickers_risk_parity = ['SPY', 'FEZ', '^N225', 'LQDH', 'HYGH', 'EMB', 'IEI', 'IEF', 'TLT', 'FXE', 'FXY', 'FXF', 'TIP', 'VMBS', 'CMBS', 'GLD', 'USO']
halflife = 126 # Set half-life for exponential weighting (6 months ~ 126 trading days)
min_periods = 60  # Minimum number of data points required

PORTFOLIOS = OrderedDict({
    "Client": {
        "function_to_call": "FIXED",
        # "ticker_subset": ['MWTIX',  'SHY'],
        "ticker_subset": x_tickers_risk_parity,
        "other_options": {
            "fixed_weights_data": fixed_weights_data,
        },
    },
    # Include other portfolios as before
    "Vol_wtd_ptfl": {
        "function_to_call": "VW",
        # "ticker_subset": [x for x in x_tickers if x not in ['SHY']],
        "ticker_subset": x_tickers_risk_parity,
        "other_options": {
            "halflife": halflife,
            "min_periods": min_periods,
        },
    },
    "EW_ptfl": {
        "function_to_call": "EW",
        "ticker_subset": x_tickers,
    },
    "Tracking_ptfl": {
        "function_to_call": "TRACK",
        "ticker_subset": x_tickers_risk_parity,
        "other_options": {
            "target_portfolio_name": 'Client',
            # Initial weights will be set dynamically
        },
    },
    "Tracking_ptfl_penalty": {  # New tracking portfolio with penalty
        "function_to_call": "TRACK_PENALTY",
        "ticker_subset": x_tickers_risk_parity,
        "other_options": {
            "target_portfolio_name": 'Client',
            "penalty_weight": 1e0,  # Set penalty parameter
            # Initial weights will be set dynamically
        },
    },
})
