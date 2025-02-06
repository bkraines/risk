# import streamlit as st
# from util import format_date
# from data import build_factor_data2
# from market_feedback import draw_market_feedback_scatter

# halflifes = [21, 63, 126, 252]

# @st.cache_data
# def build_factor_data_with_cache(halflifes):
#     return build_factor_data2(halflifes)

# factor_data = build_factor_data_with_cache(halflifes)
# factor_list = factor_data['factor_name'].values

# corr_asset   = st.selectbox('Correlation Asset', options=factor_list, index=1)
# return_start = st.date_input('Start', value='2024-12-31') #, on_change)
# return_end   = st.date_input('End', value='today')
# vol_type     = st.selectbox('Volatility Halflife', options=halflifes, index=1)
# corr_type    = st.selectbox('Correlation Halflife', options=halflifes, index=1)
# return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'

# fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)

# st.write(fig)


### TEST 3

import streamlit as st
import plotly.express as px
import pandas as pd
import yfinance as yf

data = yf.download('SPY', start='2020-01-01', end='2023-01-01')['Close']
fig = px.line(data)
# fig.show()
st.write(fig)


### TEST 2

# import streamlit as st

# import plotly.express as px
# import pandas as pd
# import numpy as np

# # Generate sample data
# np.random.seed(42)
# data = {
#     'x': np.random.randn(50),
#     'y': np.random.randn(50),
#     'category': np.random.choice(['A', 'B', 'C'], size=50)
# }
# df = pd.DataFrame(data)

# # Create scatter plot
# fig = px.scatter(
#     df, 
#     x='x', 
#     y='y', 
#     color='category', 
#     title='Sample Scatter Plot with Plotly',
#     labels={'x': 'X Axis', 'y': 'Y Axis'}
# )

# # Show plot
# # fig.show()
# st.write(fig)


#### TEST 1

# # st.write('hello world ttt')
