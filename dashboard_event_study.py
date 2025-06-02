from typing import Hashable
import streamlit as st

from risk_config import EVENT_STUDIES
from risk_data import get_factor_data
from risk_event_study import draw_event_study

from dashboard_interface import add_sidebar_defaults


def choose_date_range(before_default: int, after_default: int, step:int):
    col1, col2 = st.columns([1, 1])
    with col1:
        before = st.number_input('Days before event', min_value=0, value=before_default, step=step)
    with col2:
        after = st.number_input('Days after event', min_value=0, value=after_default, step=step)
    return before, after


def choose_event_study(event_studies):
    event_study_name = st.selectbox('Event Study', options=event_studies.keys(), index=0)

    # Check if the event name changed since last run
    if ('prior_event_name' not in st.session_state) or (st.session_state.prior_event_name != event_study_name):
        event_pair = event_studies[event_study_name]
        st.session_state.num_pairs = len(event_pair)
        st.session_state.prior_event_name = event_study_name

    # Fallback if not set
    if 'num_pairs' not in st.session_state:
        st.session_state.num_pairs = len(event_studies[event_study_name])

    return event_studies[event_study_name]


def build_event_list(factor_data, event_pairs):
    factor_list   = factor_data.indexes['factor_name']
    earliest_date = factor_data.indexes['date'].min().date()
    latest_date   = factor_data.indexes['date'].max().date()

    event_list = []
    for i in range(st.session_state.num_pairs):
        predefined_event = (i < len(event_pairs))
        col1, col2 = st.columns([1, 1])
        factor = col1.selectbox(label='Factor' if i == 0 else '',
                                options=factor_list,
                                index=(list(factor_list).index(event_pairs[i][0])
                                        if predefined_event else i),
                                key=f'factor_{i}',
                                label_visibility='collapsed')
        event_date = col2.date_input(label='Event Date' if i==0 else '',
                                     value=(event_pairs[i][1]
                                            if predefined_event else latest_date),
                                     min_value=earliest_date,
                                     max_value=latest_date,
                                     key=f'date_{i}',
                                     label_visibility='collapsed')

        if (factor, event_date) not in event_list:
            event_list.append((factor, event_date))
        else:
            st.warning(f"Duplicate pair ({factor}, {event_date}) skipped.")
                        
    def add_pair():
        st.session_state.num_pairs += 1

    def remove_pair():
        if st.session_state.num_pairs > 1:
            st.session_state.num_pairs -= 1

    col_add, col_remove, _, _ = st.columns(4, gap='small')
    with col_add:
        st.button('[ + ]', key='add_pair', on_click=add_pair)
    with col_remove:
        st.button('[ – ]', key='remove_pair', on_click=remove_pair,
                  disabled=(st.session_state.num_pairs == 1))

    return event_list


def build_dashboard(factor_data):
    # TODO:  Include 'before' and 'after' arguments in `EVENT_STUDIES` dict
    # FIXME: Uncentered data should return level, not cumulative return
    with st.sidebar:
        event_pairs = choose_event_study(EVENT_STUDIES)
        event_list = build_event_list(factor_data, event_pairs)
        before, after  = choose_date_range(before_default=63, after_default=252, step=21)
        reverse_y_axis = st.toggle('Reverse y-axis', key='reverse_y_axis', value=False)
        center_y_axis  = st.toggle('Center y-axis', key='center_y_axis', value=True)

    ret_df = factor_data.ret.to_pandas()
    fig = draw_event_study(ret_df, 
                           event_list, 
                           before=before, 
                           after=after, 
                           reverse_y_axis=reverse_y_axis, 
                           center_y_axis=center_y_axis)
    st.write(fig)

    add_sidebar_defaults()


if __name__ == '__main__':
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
