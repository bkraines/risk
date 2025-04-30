from typing import Optional
import streamlit as st
import xarray as xr

from risk_lib.data import get_factor_master, get_portfolios
from dashboard.interface import add_sidebar_defaults
from risk_lib.util import move_columns_to_front


def add_hyperlinks():
    links = {'Daily Shot': 'https://thedailyshot.com/',
             'US Yield Curve': 'https://www.ustreasuryyieldcurve.com/charts/treasuries-time-series',
             'Nishant Kumar': 'https://x.com/nishantkumar07',
             'Cliff Asnes': 'https://www.aqr.com/Insights/Perspectives'}
    for name, url in links.items():
        st.markdown(f"[{name}]({url})")


def build_dashboard(factor_data: Optional[xr.Dataset] = None):
    first_columns = ['description', 'diffusion_type', 'multiplier']
    factor_master = get_factor_master(factor_data).pipe(move_columns_to_front, first_columns)
    portfolios = get_portfolios().unstack().rename('weight').to_frame()[lambda x: x != 0].dropna()
    
    st.dataframe(portfolios, height=500)
    st.dataframe(factor_master, height=3000)
    add_hyperlinks()
    add_sidebar_defaults()


if __name__ == "__main__":
    # factor_data = get_factor_data()
    build_dashboard()
    # add_sidebar_defaults()
    # del(factor_data)
    
    