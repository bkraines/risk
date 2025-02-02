import streamlit as st

from util import format_date
from data import build_factor_data2
from market_feedback import draw_market_feedback_scatter

halflifes = [21, 63, 126, 252]

@st.cache_data
def build_factor_data_with_cache(halflifes):
    return build_factor_data2(halflifes)


factor_data = build_factor_data_with_cache(halflifes)
factor_list = factor_data['factor_name'].values

# return_start = '2024-12-31'
# return_end = '2025-01-28'
# return_title = f'Month-to-Date Return (std, as of {format_date(return_end)})'


# date_latest = factor_data['date'].max().values[0]

corr_asset = st.selectbox('Correlation Asset', options=factor_list, index=1)

return_start = st.date_input('Start', value='2024-12-31') #, on_change)
return_end = st.date_input('End', value='today')
return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'
vol_type = st.selectbox('Volatility Halflife', options=halflifes, index=1)
corr_type = st.selectbox('Correlation Halflife', options=halflifes, index=1)


# return_start = '2025-01-21'
# return_end = '2025-01-28'
# return_title = f'5-day Return (std, {format_date(return_start)} - {format_date(return_end)})'

# vol_type = 63
# corr_type = 63
# corr_asset = 'SPY'

fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)
# fig.show() #renderer='png')
# fig.write_html('fed_feedback.html')
# fig.write_image('fed_feedback.png')

st.write(fig)