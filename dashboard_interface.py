from datetime import date, datetime
from typing import Iterable, Optional
import streamlit as st

import os
from risk_data import get_factor_data
from risk_dates import build_date_options
from risk_util import check_memory_usage, summarize_memory_usage, get_directory_last_updated_time
from risk_config import CACHE_TARGET, MARKET_EVENTS, ROLLING_WINDOWS, CACHE_DIR, CACHE_FILENAME


def force_data_refresh():
    with st.sidebar:
        if st.button('Refresh Data'):
            get_factor_data(read_cache=False)
        st.write(f'Cache target: {CACHE_TARGET}')
        if CACHE_TARGET == 'disk':
            last_updated = get_directory_last_updated_time(os.path.join(CACHE_DIR, CACHE_FILENAME))
            st.write(f'Last updated: {last_updated:%Y-%m-%d %H:%M:%S}')


def display_memory_usage():
    with st.sidebar:
        st.write(f'Memory usage: {check_memory_usage()} MB')
        st.table(summarize_memory_usage())


def add_sidebar_defaults():
    force_data_refresh()
    display_memory_usage()


if __name__ == "__main__":
    add_sidebar_defaults()


def select_date_range(
    date_list: Iterable[datetime],
    rolling_windows: Optional[dict[str, int]] = ROLLING_WINDOWS,
    market_events: Optional[dict[str, tuple[datetime, datetime]]] = MARKET_EVENTS,
    default_option: Optional[str] = None,
) -> tuple[date, date]:
    """
    Display a Streamlit UI to select a date range from trailing windows, named ranges, to-date periods, or a custom picker.

    Parameters
    ----------
    date_list : Iterable[datetime]
        A sorted iterable of available dates.
    trailing_windows : Optional[dict[str, int]]
        Mapping of trailing window labels to their respective lengths (e.g., "1y": 252).
    market_events : Optional[dict[str, tuple[datetime, datetime]]]
        Mapping of named event labels to specific (start, end) date tuples.
    to_date_windows : Optional[dict[str, Callable[[datetime], tuple[datetime, datetime]]]]
        Mapping of to-date labels (e.g., "YTD", "MTD") to callables producing (start, end) dates.
    default_option : Optional[str]
        Preferred default selection in the dropdown. Defaults to 'max' if not found.

    Returns
    -------
    tuple[datetime, datetime]
        The selected (start_date, end_date) from the UI.
    """
    # TODO: Pass in a list of to_date_windows that we want to include 
    #       then from the dictionary from a An str-> Callable mapping.
    # TODO: Break out Market Events into separate dropdown? 
    #       Use an `optgroup` to separate the event types?
    # TODO: Allow custom rolling windows?


    def clean_default_option(default_option, date_options_names):
        if default_option and (default_option in date_options_names):
            initial_selection = default_option
        elif "max" in date_options_names:
            initial_selection = "max"
        else:
            initial_selection = date_options_names[0]
        return initial_selection    

    def initialize_custom_dates(initial_selection):
        # Gemini generated
        if 'custom_start_date' in st.session_state and 'custom_end_date' in st.session_state:
            return  # Session state already initialized

        if date_options_dict[initial_selection] is not None:
            _initial_session_start, _initial_session_end = date_options_dict[initial_selection]
        elif date_options_dict["max"] is not None:  # Fallback to "max" if initial_selection was "custom"
            _initial_session_start, _initial_session_end = date_options_dict["max"]
        else:  # Ultimate fallback
            _initial_session_start, _initial_session_end = earliest_date, latest_date

        st.session_state.custom_start_date = _initial_session_start
        st.session_state.custom_end_date = _initial_session_end


    # if not date_list:
    #     raise ValueError(f"List of dates must be nonempty")
    date_list = sorted(date_list)
    earliest_date = min(date_list)
    latest_date   = max(date_list)

    date_options_dict = build_date_options(date_list, rolling_windows, market_events)
    date_options_names = list(date_options_dict.keys())
    default_option = clean_default_option(default_option, date_options_names)
    initialize_custom_dates(default_option)

    selected_range = st.selectbox("Date Range", date_options_names, index=date_options_names.index(default_option))

    if selected_range != "custom":
        start_date, end_date = date_options_dict[selected_range]
        st.session_state.custom_start_date = start_date
        st.session_state.custom_end_date = end_date
    else:
        # Custom date range, careful that st.date_input might return a tuple or None
        start_date = st.date_input("Start date", 
                                   value=st.session_state.custom_start_date, 
                                   min_value=earliest_date, 
                                   max_value=latest_date)
        end_date   = st.date_input("End date", 
                                   value=st.session_state.custom_end_date, 
                                   min_value=start_date if start_date else earliest_date, 
                                   max_value=latest_date)
        if not isinstance(start_date, date):
            raise ValueError(f"Expected a single date, but got {start_date} of type {type(start_date)}")
        if not isinstance(end_date, date):
            raise ValueError(f"Expected a single date, but got {end_date} of type {type(end_date)}")
        if start_date > end_date:
            st.error("Start date must be before end date.")
        else:
            st.session_state.custom_start_date = start_date
            st.session_state.custom_end_date = end_date

    st.caption(f"{start_date.strftime(r'%Y-%m-%d')} to {end_date.strftime(r'%Y-%m-%d')}")

    return start_date, end_date

