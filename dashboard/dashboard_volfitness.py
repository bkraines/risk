import streamlit as st

import pandas as pd
import xarray as xr

from risk_lib.data import get_factor_data
from risk_lib.stats import get_zscore
from risk_lib.chart import draw_zscore_qq, plot_qq_df
from risk_lib.config import HALFLIFES
from dashboard.interface import select_date_range
from dashboard.interface import add_sidebar_defaults

def build_dashboard(factor_data: xr.Dataset) -> None:
    factor_list = factor_data['factor_name'].values
    with st.sidebar:
        factor_1   = st.selectbox('Factor 1', options=factor_list, index=0)
        vol_type_1 = st.selectbox('Volatility 1', options=HALFLIFES, index=1)
        factor_2   = st.selectbox('Factor 2', options=factor_list, index=1)
        # vol_type_2 = st.selectbox('Volatility 2', options=HALFLIFES, index=1)
        start_date, end_date = select_date_range(factor_data.indexes['date'], default_option='1y')
    
    factor_data['zscore'] = get_zscore(factor_data.ret, factor_data.vol)
    
    ds = factor_data['zscore'].sel(date=slice(start_date, end_date))
    df = (ds.sel(factor_name=[factor_1, factor_2], 
                 vol_type=vol_type_1,
                 date=slice(start_date, end_date),)
          .to_pandas()
          .dropna()
          )   
    
    # df = (pd.concat([ds.sel(factor_name=factor_1, vol_type=vol_type_1).to_pandas(),
    #                  ds.sel(factor_name=factor_2, vol_type=vol_type_2).to_pandas()
    #                  ], axis=1)).dropna()
    fig = plot_qq_df(df)
    st.plotly_chart(fig)
    
    
    # figs = {'fig_1': draw_zscore_qq(ds.ret, ds.vol, factor_name=factor_1, vol_type=vol_type_1),
    #         'fig_2': draw_zscore_qq(ds.ret, ds.vol, factor_name=factor_2, vol_type=vol_type_2)}
    
    # for fig in figs.values():
    #     st.plotly_chart(fig)
        
    add_sidebar_defaults()
    

if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
