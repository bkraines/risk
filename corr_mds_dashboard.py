import streamlit as st

from config import STREAMLIT_CACHE, HALFLIFES
from data import get_factor_data, build_factor_data
from corr_mds import run_mds
from util import check_memory_usage, summarize_memory_usage

def build_streamlit_dashboard(factor_data):
    args = {'random_state': 42, 
            'n_init': 100}
    dates = factor_data.date.sel(date=slice('2020', None)).values
    fig = run_mds(factor_data.resample(date='W').last(), 
                  transformation='rotate_initial', 
                  dates=dates,
                  start_date='2020', 
                  tick_range=1,
                  animate=True,
                  drop_composites=True,
                  drop_trump=False,
                  **args)
    st.write(fig)

    with st.sidebar:
        st.write(f'Memory usage: {check_memory_usage()} MB')
        st.table(summarize_memory_usage())


if __name__ == "__main__":
    factor_data = build_factor_data(HALFLIFES)
    build_streamlit_dashboard(factor_data)
    del(factor_data)