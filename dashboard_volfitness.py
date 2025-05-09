import streamlit as st

from typing import Any

import xarray as xr

from risk_data import get_factor_data
from risk_chart import draw_zscore_qq
from risk_config import HALFLIFES
from dashboard_interface import select_date_range
from dashboard_interface import add_sidebar_defaults

def build_dashboard(factor_data: xr.Dataset) -> None:
    factor_list = factor_data['factor_name'].values
    with st.sidebar:
        # TODO: Allow selection of arbitrary number of factors
        factor_1   = st.selectbox('Factor 1', options=factor_list, index=0)
        vol_type_1 = st.selectbox('Volatility 1', options=HALFLIFES, index=1)
        factor_2   = st.selectbox('Factor 2', options=factor_list, index=1)
        vol_type_2 = st.selectbox('Volatility 2', options=HALFLIFES, index=1)
        start_date, end_date = select_date_range(factor_data.indexes['date'], default_option='max')
    
    qq_series: list[tuple[str, Any]] = [(factor_1, vol_type_1), 
                                        (factor_2, vol_type_2)]
    da = factor_data.sel(date=slice(start_date, end_date))
    fig = draw_zscore_qq(da.ret, da.vol, qq_series)
    
    st.plotly_chart(fig)
    st.write('(Observed values occur with frequency of theoretical values.)')
    
    add_sidebar_defaults()    


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
