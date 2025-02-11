# TEST 9 - NO CACHE

import streamlit as st
from util import format_date
from data import build_factor_data2
from market_feedback import draw_market_feedback_scatter

halflifes = [21, 63] #, 126, 252]

@st.cache_data
def build_factor_data_with_cache(halflifes):
    return build_factor_data2(halflifes)

factor_data = build_factor_data_with_cache(halflifes)
factor_list = factor_data['factor_name'].values

corr_asset   = st.selectbox('Correlation Asset', options=factor_list, index=1)
return_start = st.date_input('Start', value='2024-12-31') #, on_change)
return_end   = st.date_input('End', value='today')
vol_type     = st.selectbox('Volatility Halflife', options=halflifes, index=1)
corr_type    = st.selectbox('Correlation Halflife', options=halflifes, index=1)
return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'

fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)

st.write(fig)

# TEST 8 - MULTIPLE DROPDOWNS

# import streamlit as st
# from data import build_factor_data2
# from market_feedback import draw_market_feedback_scatter

# halflifes = [63, 252]
# factor_data = build_factor_data2(halflifes)

# return_start = st.date_input('Start', value='2024-12-31') #, on_change)
# return_end   = st.date_input('End', value='today')
# vol_type     = st.selectbox('Volatility Halflife', options=halflifes, index=1)
# corr_type    = st.selectbox('Correlation Halflife', options=halflifes, index=1)
# # return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'
# return_title = f'Returns (std)'

# factor_list = factor_data['factor_name'].values
# corr_asset = st.selectbox('Correlation Asset', options=factor_list, index=1)

# fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)
# st.write(fig)


# TEST 7 - SINGLE DROP DOWN

# import streamlit as st
# from data import build_factor_data2
# from market_feedback import draw_market_feedback_scatter

# halflifes = [63, 252]
# factor_data = build_factor_data2(halflifes)
# df = factor_data.cret.to_pandas()['SPY'].sort_values(ascending=False).head(10)
# st.write(df)

# return_start = '2024-12-31'
# return_end = factor_data.indexes['date'].max()
# vol_type   = 63
# corr_type  = 63
# corr_asset = 'SPY'
# return_title = f'Returns (std)'

# factor_list = factor_data['factor_name'].values
# corr_asset = st.selectbox('Correlation Asset', options=factor_list, index=1)

# fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)
# st.write(fig)



## TEST 6 - USE FACTOR DATA

# import streamlit as st
# from data import build_factor_data2
# from market_feedback import draw_market_feedback_scatter

# halflifes = [63, 252]
# factor_data = build_factor_data2(halflifes)
# df = factor_data.cret.to_pandas()['SPY'].sort_values(ascending=False).head(10)
# st.write(df)

# return_start = '2024-12-31'
# return_end = factor_data.indexes['date'].max()
# vol_type   = 63
# corr_type  = 63
# corr_asset = 'SPY'
# return_title = f'Returns (std)'

# fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)
# st.write(fig)


## TEST 5 - USE FACTOR DATA

# import streamlit as st
# from data import build_factor_data2

# halflifes = [63, 252]
# ds = build_factor_data2(halflifes)
# df = ds.cret.to_pandas()['SPY'].sort_values(ascending=False).head(10)
# st.write(df)


## TEST 4 - BIGGER YAHOO CHART

# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import yfinance as yf
# from data import get_factor_master

# factor_master = get_factor_master()
# factor_list = factor_master.index.values.tolist()
# data = yf.download(factor_list, start='2020-01-01', end='2023-01-01')['Close']
# fig = px.line(data)
# # fig.show()
# st.write(fig)





### TEST 3 - YAHOO CHART

# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import yfinance as yf

# data = yf.download('SPY', start='2020-01-01', end='2023-01-01')['Close']
# fig = px.line(data)
# # fig.show()
# st.write(fig)


### TEST 2 - LINE CHART

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


#### TEST 1 - HELLO WORLD

# # st.write('hello world ttt')
