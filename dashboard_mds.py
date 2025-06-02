import streamlit as st

from risk_data import get_factor_data
from risk_corr_mds import run_mds
from dashboard_interface import add_sidebar_defaults

def build_dashboard(factor_data):
    args = {'random_state': 42, 
            'n_init': 100}
    
    with st.sidebar:
        earliest_date = factor_data.indexes['date'].min().date()
        latest_date = factor_data.indexes['date'].max().date()
        end_date = st.date_input("End date", latest_date, earliest_date, latest_date)
        animate    = st.toggle('Animate',    value=False)
        composites = st.toggle('Composites', value=True)
        trump      = st.toggle('Trump',      value=True)
        portfolios = st.toggle('Portfolios', value=True)
    
    if animate:
        ds = (factor_data
              .sel(date=slice('2020', latest_date))
              .resample(date='W').last())
        dates = ds.date.values
        start_date = '2020'
    else:
        ds = factor_data.sel(date=slice(None, end_date))
        dates = (ds.date.values[[-63, -42, -21, -1]]
                 .astype('datetime64[D]').astype(str))
        start_date = dates[0]
        
    
    fig = (run_mds(ds, 
            transformation='rotate_initial', 
            dates=dates,
            start_date=start_date, 
            tick_range=1,
            animate=animate,
            drop_composites=not(composites),
            drop_trump=not(trump),
            drop_portfolios=not(portfolios),
            **args))

    st.write(fig)

    add_sidebar_defaults()


if __name__ == '__main__':
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
