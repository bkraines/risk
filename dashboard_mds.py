import streamlit as st
from datetime import datetime
from risk_config import HALFLIFES
from risk_data import get_factor_data
from risk_corr_mds import run_mds
from dashboard_interface import add_sidebar_defaults


def build_dashboard(factor_data):
    # TODO: If animate, replace `end_date` selector with `select_date_window`,
    #       with dropdown for sampling (rebalancing?) frequency
    #       see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    
    args = {'random_state': 42, 
            'n_init': 100}

    model_options = HALFLIFES
    model_default = model_options.index(63) if 63 in model_options else 0
    
    default_date = datetime(2024, 11, 29)
    with st.sidebar:
        earliest_date = factor_data.indexes['date'].min().date()
        latest_date   = factor_data.indexes['date'].max().date()
        end_date      = st.date_input("End date", default_date, earliest_date, latest_date)
        corr_type     = st.selectbox('Correlation Halflife', options=model_options, index=model_default)
        animate       = st.toggle('Animate', value=False)
        composites    = st.toggle('Include Themes', value=False)
        election      = st.toggle('Include Election', value=False)
        portfolios    = st.toggle('Include Portfolios', value=False)
    
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
            drop_election=not(election),
            drop_portfolios=not(portfolios),
            corr_type=corr_type,
            **args))

    st.write(fig)

    add_sidebar_defaults()


if __name__ == '__main__':
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
