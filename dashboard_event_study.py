from typing import Hashable
import streamlit as st

from risk_config import EVENT_STUDIES
from risk_data import get_factor_data
from risk_event_study import draw_event_study

from dashboard_interface import add_sidebar_defaults


def build_dashboard(factor_data):
    # TODO: Include predefined event studies
    factor_list   = factor_data.indexes['factor_name']
    earliest_date = factor_data.indexes['date'].min().date()
    latest_date   = factor_data.indexes['date'].max().date() #.sel(date=slice(None, '2024'))

    # initial_pair = [('SPY', '2018-01-28'), ('SPY', '2025-04-01')] # Align VIX peak 1w prior (SPY)
    # initial_pair = EVENT_PAIRS['hi_vix3']
    # st.session_state.numpairs = len(initial_pair)
    # if 'num_pairs' not in st.session_state:
    #     st.session_state.num_pairs = len(initial_pair)
    def add_pair():
        st.session_state.num_pairs += 1
    def remove_pair():
        if st.session_state.num_pairs > 1:
            st.session_state.num_pairs -= 1
    
    event_list = []
    with st.sidebar:

        # Track the selected event name
        event_study_name = st.selectbox("Event Study", options=EVENT_STUDIES.keys(), index=0)

        # Check if the event name changed since last run
        if ('prior_event_name' not in st.session_state) or (st.session_state.prior_event_name != event_study_name):
            event_pair = EVENT_STUDIES[event_study_name]
            st.session_state.num_pairs = len(event_pair)
            st.session_state.prior_event_name = event_study_name  # update tracker

        # fallback if not set
        if 'num_pairs' not in st.session_state:
            st.session_state.num_pairs = len(EVENT_STUDIES[event_study_name])

        event_pair = EVENT_STUDIES[event_study_name]  # get fresh list

        for i in range(1, st.session_state.num_pairs + 1):

            col1, col2 = st.columns([1, 1])
            if i < len(event_pair)+1: # Choose the next pair from the event list
                factor      = col1.selectbox(label='Factor' if i == 1 else '',
                                             options=factor_list,
                                             index=list(factor_list).index(event_pair[i-1][0]),
                                             key=f'factor_{i}',
                                            #  label_visibility='visible' if i == 1 else 'collapsed',
                                            label_visibility='collapsed',)
                event_date = col2.date_input(label='Event Date'  if i==1 else '',
                                             value=event_pair[i-1][1], 
                                             min_value=earliest_date, 
                                             max_value=latest_date, 
                                             key=f'date_{i}', 
                                            #  label_visibility='visible' if i == 1 else 'collapsed'
                                             label_visibility='collapsed')
            else: # Choose the next factor from the factor list, use the latest date
                factor = col1.selectbox(label=f'Factor {i}', options=factor_list, index=i-1, key=f'factor_{i}', label_visibility='collapsed')
                event_date = col2.date_input(label=f'Date {i}', value=latest_date, min_value=earliest_date, max_value=latest_date, key=f'date_{i}', label_visibility='collapsed')

            if (factor, event_date) not in event_list:
                event_list.append((factor, event_date))
            else:
                st.warning(f"Duplicate pair ({factor}, {event_date}) skipped.")

        col_add, col_remove, _, _ = st.columns(4, gap='small')
        with col_add:
            st.button("[ + ]", key="add_pair", on_click=add_pair)
        with col_remove:
            st.button("[ – ]", key="remove_pair", on_click=remove_pair, disabled=(st.session_state.num_pairs == 1))

        col1, col2 = st.columns([1, 1])
        with col1:
            before = st.number_input("Days before event", min_value=0, value=63, step=21)
        with col2:
            after = st.number_input("Days after event", min_value=0, value=252, step=21)

    ret_df = factor_data.ret.to_pandas()
    fig = draw_event_study(ret_df, event_list, before=before, after=after)
    st.write(fig)

    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
