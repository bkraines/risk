import streamlit as st

from risk_lib.data import get_factor_data
from risk_lib.chart import draw_volatility, draw_correlation, draw_cumulative_return, draw_volatility_ratio, draw_beta
from risk_lib.config import HALFLIFES
from dashboard.interface import select_date_range
from dashboard.interface import add_sidebar_defaults

# TODO: Add vol ratio, add beta

def build_dashboard(factor_data):
    factor_list = factor_data['factor_name'].values

    with st.sidebar:
        factor_1 = st.selectbox('Factor 1', options=factor_list, index=0)
        factor_2 = st.selectbox('Factor 2', options=factor_list, index=1)
        # start_date = st.date_input('Start Date', value='2023-12-31')
        # end_date = st.date_input('End Date', value=date_latest)
        # start_date, end_date = get_date_picker(default_start=factor_data.indexes['date'].min().date())
        start_date, end_date = select_date_range(factor_data.indexes['date'], default_option='1y')

    ds = factor_data.sel(date=slice(start_date, end_date))
    figs = {'cret':  draw_cumulative_return(ds.cret, factor_name=factor_1, factor_name_1=factor_2),
            'corr':  draw_correlation(ds.corr, factor_name=factor_1, factor_name_1=factor_2, corr_type=HALFLIFES),
            'beta':  draw_beta(ds, factor_name=factor_1, factor_name_1=factor_2),
            'vol_1': draw_volatility(ds.vol, factor_name=factor_1, vol_type=HALFLIFES),
            'vol_2': draw_volatility(ds.vol, factor_name=factor_2, vol_type=HALFLIFES),
            'vol_ratio': draw_volatility_ratio(ds.vol, factor_name=factor_1, factor_name_1=factor_2, vol_type=HALFLIFES),
            }

    for fig in figs.values():
        st.write(fig)

    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
