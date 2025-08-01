import streamlit as st

from risk_data import get_factor_data
from risk_market_feedback import draw_market_feedback_scatter
from risk_config import HALFLIFES
from dashboard_interface import add_sidebar_defaults, select_date_window

def build_dashboard(factor_data):
    # TODO: Add peak memory usage (before deleting factor_data)
    # TODO: Aḍd initial memory usage (before loading factor_data)
    factor_list = factor_data['factor_name'].values

    model_options = HALFLIFES
    model_default = model_options.index(63) if 63 in model_options else 0

    with st.sidebar:
        corr_asset   = st.selectbox('Correlation Asset', options=factor_list, index=0)
        return_start, return_end = select_date_window(factor_data.indexes['date'], default_window_name='1m')
        col1, col2 = st.columns([1, 1])
        vol_type     = col1.selectbox('Volatility Halflife', options=model_options, index=model_default)
        corr_type    = col2.selectbox('Correlation Halflife', options=model_options, index=model_default)
        include_themes     = st.toggle('Include Themes',     value=True)
        include_portfolios = st.toggle('Include Portfolios', value=False)

    exclude = []
    if not include_themes:
        exclude.append('Theme')
    if not include_portfolios:
        exclude.append('Portfolio')

    return_title = None #f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'
    fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title, exclude)

    st.write(fig)
    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
