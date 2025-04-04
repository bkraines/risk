import streamlit as st

from data import get_factor_data
from chart import draw_volatility, draw_correlation, draw_cumulative_return
from util import check_memory_usage, summarize_memory_usage
from config import HALFLIFES

from datetime import datetime, timedelta

def get_date_picker(default_start=datetime(2000, 1, 1)):
    """
    Displays a Streamlit date range selector with standard presets.
    
    Args:
        default_start (datetime): The earliest available date (used for "max" option).

    Returns:
        (datetime, datetime): A tuple containing the selected start and end dates.
    """
    today = datetime.today()

    # Preset ranges
    date_options = {
        "1d": (today - timedelta(days=1), today),
        "5d": (today - timedelta(days=5), today),
        "1m": (today - timedelta(days=30), today),
        "1y": (today - timedelta(days=365), today),
        "5y": (today - timedelta(days=5 * 365), today),
        "max": (default_start, today),
        "custom": None,
    }

    selected_range = st.selectbox("Select date range", list(date_options.keys()))

    if selected_range != "custom":
        start_date, end_date = date_options[selected_range]
        st.write(f"Selected date range: **{start_date.date()} to {end_date.date()}**")
    else:
        start_date = st.date_input("Start date", today - timedelta(days=30))
        end_date = st.date_input("End date", today)
        if start_date > end_date:
            st.error("Start date must be before end date.")
        st.write(f"Custom date range: **{start_date} to {end_date}**")

    return start_date, end_date


def build_dashboard_vol(factor_data):
    factor_list = factor_data['factor_name'].values
    date_latest = factor_data.indexes['date'].max().date()

    with st.sidebar:
        factor_1 = st.selectbox('Factor 1', options=factor_list, index=1)
        factor_2 = st.selectbox('Factor 2', options=factor_list, index=2)
        # start_date = st.date_input('Start Date', value='2023-12-31')
        # end_date = st.date_input('End Date', value=date_latest)
        start_date, end_date = get_date_picker(default_start=factor_data.indexes['date'].min().date())

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
