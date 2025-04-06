import streamlit as st

from data import get_factor_data
from dates import format_date, select_date_range
from market_feedback import draw_market_feedback_scatter
from config import HALFLIFES
from util import check_memory_usage, summarize_memory_usage

def build_streamlit_dashboard(factor_data):
    # TODO: Add peak memory usage (before deleting factor_data)
    # TODO: Aḍd initial memory usage (before loading factor_data)
    factor_list = factor_data['factor_name'].values

    with st.sidebar:
        corr_asset   = st.selectbox('Correlation Asset', options=factor_list, index=0)
        return_start, return_end = select_date_range(factor_data.indexes['date'])
        # return_start = st.date_input('Start', value='2024-12-31') #, on_change)
        # return_end   = st.date_input('End', value='today')
        vol_type     = st.selectbox('Volatility Halflife', options=HALFLIFES, index=0)
        corr_type    = st.selectbox('Correlation Halflife', options=HALFLIFES, index=0)

    return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'
    fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)

    st.write(fig)

    with st.sidebar:
        st.write(f'Memory usage: {check_memory_usage()} MB')
        st.table(summarize_memory_usage())


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_streamlit_dashboard(factor_data)
    del(factor_data)
