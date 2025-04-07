import streamlit as st

from data import get_factor_data
from corr_mds import run_mds
from util import check_memory_usage, summarize_memory_usage

def build_streamlit_dashboard(factor_data):
    args = {'random_state': 42, 
            'n_init': 100}
    
    with st.sidebar:
        animate = st.checkbox('Animate', value=False)
        composites = st.checkbox('Composites', value=False)
        trump = st.checkbox('Trump', value=False)
    
    if animate:
        dates = factor_data.date.sel(date=slice('2020', None)).values
        start_date = '2020'
        ds = factor_data.resample(date='W').last()
    else:
        dates = factor_data.date.values[[-63, -42, -21, -1]].astype('datetime64[D]').astype(str)
        start_date = '2025'
        ds = factor_data
    
    fig = (run_mds(ds, 
            transformation='rotate_initial', 
            dates=dates,
            start_date=start_date, 
            tick_range=1,
            animate=animate,
            drop_composites=not(composites),
            drop_trump=not(trump),
            **args))

    # fig = run_mds(ds, 
    #               transformation='rotate_initial', 
    #               dates=dates,
    #               start_date='2020', 
    #               tick_range=1,
    #               animate=True,
    #               drop_composites=True,
    #               drop_trump=False,
    #               **args)
    
    # dates = factor_data.date.sel(date=slice('2020', None)).values
    # fig = run_mds(factor_data.resample(date='W').last(), 
    #               transformation='rotate_initial', 
    #               dates=dates,
    #               start_date='2020', 
    #               tick_range=1,
    #               animate=True,
    #               drop_composites=True,
    #               drop_trump=False,
    #               **args)
    st.write(fig)

    with st.sidebar:
        st.write(f'Memory usage: {check_memory_usage()} MB')
        st.table(summarize_memory_usage())


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_streamlit_dashboard(factor_data)
    del(factor_data)