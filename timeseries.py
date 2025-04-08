import streamlit as st

from data import get_factor_data
from dates import select_date_range
from chart import draw_volatility, draw_correlation, draw_cumulative_return
from util import check_memory_usage, summarize_memory_usage
from config import HALFLIFES


def build_dashboard_vol(factor_data):
    factor_list = factor_data['factor_name'].values

    with st.sidebar:
        factor_1 = st.selectbox('Factor 1', options=factor_list, index=1)
        factor_2 = st.selectbox('Factor 2', options=factor_list, index=2)
        # start_date = st.date_input('Start Date', value='2023-12-31')
        # end_date = st.date_input('End Date', value=date_latest)
        # start_date, end_date = get_date_picker(default_start=factor_data.indexes['date'].min().date())
        start_date, end_date = select_date_range(factor_data.indexes['date'], default_option='1y')

    ds = factor_data.sel(date=slice(start_date, end_date))
    figs = {'cret':  draw_cumulative_return(ds.cret, factor_name=factor_1, factor_name_1=factor_2),
            'corr':  draw_correlation(ds.corr, factor_name=factor_1, factor_name_1=factor_2, corr_type=HALFLIFES),
            'vol_1': draw_volatility(ds.vol, factor_name=factor_1, vol_type=HALFLIFES),
            'vol_2': draw_volatility(ds.vol, factor_name=factor_2, vol_type=HALFLIFES),
            }

    for fig in figs.values():
        st.write(fig)

    with st.sidebar:
        st.write(f'Memory usage: {check_memory_usage()} MB')
        st.table(summarize_memory_usage())


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard_vol(factor_data)
    del(factor_data)
