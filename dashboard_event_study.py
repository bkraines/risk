from typing import Hashable
import streamlit as st

from risk_config import EVENT_PAIRS
from risk_data import get_factor_data
from risk_event_study import draw_event_study

from dashboard_interface import add_sidebar_defaults


def build_dashboard(factor_data):
    # TODO: Include predefined event studies
    factor_list   = factor_data['factor_name'].values
    earliest_date = factor_data.indexes['date'].min().date()
    latest_date   = factor_data.indexes['date'].max().date() #.sel(date=slice(None, '2024'))

    # initial_pair = [('SPY', '2018-01-28'), ('SPY', '2025-04-01')] # Align VIX peak 1w prior (SPY)
    initial_pair = EVENT_PAIRS['hi_vix3']

    if 'num_pairs' not in st.session_state:
        st.session_state.num_pairs = 2
    def add_pair():
        st.session_state.num_pairs += 1
    def remove_pair():
        if st.session_state.num_pairs > 1:
            st.session_state.num_pairs -= 1
    
    event_list = []
    with st.sidebar:
        st.write("Events:")
        for i in range(1, st.session_state.num_pairs + 1):
            
            col1, col2 = st.columns([1, 1]) 
            with col1:
                # factor = st.selectbox(f'Factor {i}', options=factor_list, index=i-1, key=f'factor_{i}', label_visibility='collapsed')
                factor = st.selectbox(f'Factor {i}', options=factor_list, index=0, key=f'factor_{i}', label_visibility='collapsed')
            with col2:
                # event_date = st.date_input(f'Date {i}', latest_date, earliest_date, latest_date, key=f'date_{i}', label_visibility='collapsed')
                event_date = st.date_input(f'Date {i}', initial_pair[i-1][1], earliest_date, latest_date, key=f'date_{i}', label_visibility='collapsed')
            
            if (factor, event_date) not in event_list:
                event_list.append((factor, event_date))
            else:
                st.warning(f"Duplicate pair ({factor}, {event_date}) skipped.")

        col_add, col_remove, _, _ = st.columns(4, gap='small')
        with col_add:
            st.button("[ + ]", key="add_pair", on_click=add_pair)
        with col_remove:
            st.button("[ – ]", key="remove_pair", on_click=remove_pair, disabled=(st.session_state.num_pairs == 1))

        # Time window inputs
        col1, col2 = st.columns([1, 1])
        with col1:
            before = st.number_input("Days before event", min_value=0, value=63)
        with col2:
            after = st.number_input("Days after event", min_value=0, value=252)

    ret_df = factor_data.ret.to_pandas()
    fig = draw_event_study(ret_df, event_list, before=before, after=after)
    st.write(fig)

    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
