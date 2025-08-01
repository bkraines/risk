from typing import Optional
import streamlit as st
import xarray as xr

from risk_data import get_factor_master, get_factor_composites
from dashboard_interface import add_sidebar_defaults
from risk_util import move_columns_to_front


def add_hyperlinks():
    links = {'Daily Shot': 'https://thedailyshot.com/',
             'US Yield Curve': 'https://www.ustreasuryyieldcurve.com/charts/treasuries-time-series',
             'Nishant Kumar': 'https://x.com/nishantkumar07',
             'Cliff Asnes': 'https://www.aqr.com/Insights/Perspectives',
             'Policy Uncertainty': 'https://www.policyuncertainty.com/'}
    for name, url in links.items():
        st.markdown(f"[{name}]({url})")


def build_dashboard(factor_data: Optional[xr.Dataset] = None):
    first_columns = ['description', 'diffusion_type', 'multiplier']
    factor_master = get_factor_master(factor_data).pipe(move_columns_to_front, first_columns)
    composites = get_factor_composites().unstack().rename('weight').to_frame()[lambda x: x != 0].dropna()
    
    st.components.v1.html(factor_data._repr_html_(), height=600, scrolling=True)
    st.dataframe(composites, height=500)
    st.dataframe(factor_master, height=3000)
    add_hyperlinks()
    add_sidebar_defaults()


if __name__ == "__main__":
    # factor_data = get_factor_data()
    build_dashboard()
    # add_sidebar_defaults()
    # del(factor_data)
    
    