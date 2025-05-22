import streamlit as st

from risk_data import get_factor_data
from risk_dates import get_start_date_n_periods_ago
from risk_chart import draw_correlation_heatmap, draw_corr_matrix
from risk_config import HALFLIFES
from dashboard_interface import select_date_window, add_sidebar_defaults

def build_dashboard(factor_data):
    # TODO: Include time series of factor ratios
    # TODO: Toggle animation frame for correlation matrices
    #       Determine resampling, interval, etc.
    # TODO: Sort correlation matrix by factor heirarchy

    corr_index = 1 if len(HALFLIFES) > 1 else 0

    factor_list = factor_data['factor_name'].values
    with st.sidebar:
        factor = st.selectbox('Factor', options=factor_list, index=0)
        start_date, end_date = select_date_window(factor_data.indexes['date'], default_window_name='15y')
        corr_type = st.selectbox('Correlation Halflife', options=HALFLIFES, index=corr_index)
        change = st.number_input('Correlation Change (days)', min_value=1, value=21)

    fig1 = draw_correlation_heatmap(factor_data.corr.sel(date=slice(start_date, end_date)), 
                                   fixed_factor=factor, 
                                   corr_type=corr_type)    
    st.plotly_chart(fig1)
    
    
    fig2 = draw_corr_matrix(factor_data.corr.sel(date=end_date),
                            corr_type=corr_type)
    st.plotly_chart(fig2)

    date_list = factor_data.sel(date=slice(None, end_date)).indexes['date']    
    chg_date = get_start_date_n_periods_ago(date_list, change)
    # chg_date = '2025-04-08'
    corr_chg = factor_data.corr.sel(date=end_date) - factor_data.corr.sel(date=chg_date)
    title = f'Correlation Change ({change}-day change, {corr_type}-day halflife)'
    fig3 = draw_corr_matrix(corr_chg, corr_type=corr_type, title=title)
    st.plotly_chart(fig3)

    add_sidebar_defaults()
