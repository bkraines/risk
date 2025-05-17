import streamlit as st

from risk_data import get_factor_data
from risk_chart import draw_correlation_heatmap
from risk_config import HALFLIFES
from dashboard_interface import select_date_window, add_sidebar_defaults

def build_dashboard(factor_data):
    # TODO: Include time series of factor ratios
    
    corr_index = 1 if len(HALFLIFES) > 1 else 0

    factor_list = factor_data['factor_name'].values
    with st.sidebar:
        factor = st.selectbox('Factor', options=factor_list, index=0)
        start_date, end_date = select_date_window(factor_data.indexes['date'], default_window_name='max')
        corr_type = st.selectbox('Correlation Halflife', options=HALFLIFES, index=corr_index)

    fig = draw_correlation_heatmap(factor_data.corr.sel(date=slice(start_date, end_date)), 
                                   fixed_factor=factor, 
                                   corr_type=corr_type)
    st.plotly_chart(fig)

    add_sidebar_defaults()
