import streamlit as st

from risk_lib.data import get_factor_data
from risk_lib.util import format_date
from risk_lib.market_feedback import draw_market_feedback_scatter
from risk_lib.config import HALFLIFES
from dashboard.interface import add_sidebar_defaults, select_date_range

def build_dashboard(factor_data):
    # TODO: Add peak memory usage (before deleting factor_data)
    # TODO: Aḍd initial memory usage (before loading factor_data)
    factor_list = factor_data['factor_name'].values

    model_options = HALFLIFES
    model_default = model_options.index(126) if 126 in model_options else 0

    with st.sidebar:
        corr_asset   = st.selectbox('Correlation Asset', options=factor_list, index=0)
        return_start, return_end = select_date_range(factor_data.indexes['date'], default_option='MTD')
        vol_type     = st.selectbox('Volatility Halflife', options=model_options, index=model_default)
        corr_type    = st.selectbox('Correlation Halflife', options=model_options, index=model_default)

    return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'
    fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)

    st.write(fig)
    
    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
