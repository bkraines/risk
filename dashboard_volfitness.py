import streamlit as st

from typing import Any

import xarray as xr

from risk_data import get_factor_data
from risk_chart import draw_zscore_qq
from risk_config import HALFLIFES
from dashboard_interface import select_date_window
from dashboard_interface import add_sidebar_defaults

def build_dashboard(factor_data: xr.Dataset) -> None:
    factor_list = factor_data['factor_name'].values
    
    if 'num_pairs' not in st.session_state:
            st.session_state.num_pairs = 2
    def add_pair():
        st.session_state.num_pairs += 1
    def remove_pair():
        if st.session_state.num_pairs > 1:
            st.session_state.num_pairs -= 1

    factor_vol_pairs: list[tuple[str, Any]] = []
    with st.sidebar:
        
        for i in range(1, st.session_state.num_pairs + 1):
            col1, col2 = st.columns([1, 1])
            factor   = col1.selectbox(label='Factor' if i == 1 else '' , 
                                      options=factor_list, index=i-1, 
                                      key=f'factor_{i}',
                                      label_visibility='collapsed' if i != 1 else 'visible')
            vol_type = col2.selectbox(label='Volatility' if i==1 else '', 
                                      options=HALFLIFES, index=1,
                                      key=f'vol_type_{i}',
                                      label_visibility='collapsed' if i != 1 else 'visible')
            factor_vol_pairs.append((factor, vol_type))

        # TODO: This duplicates logic from `dashboard_event_study.py`
        col_add, col_remove, _, _ = st.columns(4, gap='small')
        with col_add:
            st.button("[ + ]", key="add_pair", on_click=add_pair)
        with col_remove:
            st.button("[ – ]", key="remove_pair", on_click=remove_pair, disabled=(st.session_state.num_pairs == 1))

        start_date, end_date = select_date_window(factor_data.indexes['date'], default_window_name='max')
    
    # qq_series: list[tuple[str, Any]] = [(factor_1, vol_type_1), 
    #                                     (factor_2, vol_type_2)]
    da = factor_data.sel(date=slice(start_date, end_date))
    fig = draw_zscore_qq(da.ret, da.vol, factor_vol_pairs)
    
    st.plotly_chart(fig)
    st.write('(Observed values occur with frequency of theoretical values.)')
    
    add_sidebar_defaults()    


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
