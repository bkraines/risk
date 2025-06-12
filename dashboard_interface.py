from datetime import date, datetime
from typing import Iterable, Optional
import streamlit as st

import os
from risk_data import get_factor_data
from risk_dates import build_window_map
from risk_util import check_memory_usage, summarize_memory_usage, get_directory_last_updated_time
from risk_config import CACHE_TARGET, CACHE_DIR, CACHE_FILENAME, HISTORICAL_WINDOWS, ROLLING_WINDOWS


def force_data_refresh():
    # FIXME: This just updates the cache, not the variable in memory
    with st.sidebar:
        if st.button('Refresh Data'):
            with st.spinner("Constructing factors..."):
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
    # TODO: Add clear cache button
    force_data_refresh()
    display_memory_usage()


if __name__ == "__main__":
    add_sidebar_defaults()


def select_date_window(
    date_list: Iterable[datetime],
    rolling_windows: Optional[dict[str, int]] = ROLLING_WINDOWS,
    historical_windows: Optional[dict[str, tuple[datetime, datetime]]] = HISTORICAL_WINDOWS,
    default_window_name: Optional[str] = None,
) -> tuple[date, date]:
    """
    Display a Streamlit UI to select a date range from rolling windows, expanding windows, historical windows, or a custom picker.

    Parameters
    ----------
    date_list : Iterable[datetime]
        An iterable of available dates (will be sorted internally)
    rolling_windows : Optional[dict[str, int]], default=ROLLING_WINDOWS
        Mapping of rolling window labels to their respective lengths in days (e.g. {"1y": 252})
    historical_windows : Optional[dict[str, tuple[datetime, datetime]]], default=HISTORICAL_WINDOWS
        Mapping of named historical ranges (e.g. "GFC", "COVID") to specific (start, end) date tuples
    default_window_name : Optional[str], default=None
        Window to pre-select in the dropdown; falls back to 'max' if not provided or not found

    Returns
    -------
    tuple[datetime, datetime]
        The (start_date, end_date) selected by the user
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

        if window_map[initial_selection] is not None:
            _initial_session_start, _initial_session_end = window_map[initial_selection]
        elif window_map["max"] is not None:  # Fallback to "max" if initial_selection was "custom"
            _initial_session_start, _initial_session_end = window_map["max"]
        else:  # Ultimate fallback
            _initial_session_start, _initial_session_end = earliest_date, latest_date

        st.session_state.custom_start_date = _initial_session_start
        st.session_state.custom_end_date = _initial_session_end

    # TODO: Find better way to guarantee date_list is sorted and not null
    # if not date_list:
    #     raise ValueError(f"List of dates must be nonempty")
    date_list = sorted(date_list)
    earliest_date = min(date_list)
    latest_date   = max(date_list)

    window_map = build_window_map(date_list, rolling_windows, historical_windows)
    window_names = list(window_map.keys())
    default_window_name = clean_default_option(default_window_name, window_names)
    initialize_custom_dates(default_window_name)

    selected_range = st.selectbox("Date Range", window_names, index=window_names.index(default_window_name))

    # TODO: Instead ensure that window doesn't map to None?
    if selected_range != "custom":
        start_date, end_date = window_map[selected_range]
        st.session_state.custom_start_date = start_date
        st.session_state.custom_end_date = end_date
    else:
        # Select custom date range
        col1, col2 = st.columns([1, 1])
        start_date = col1.date_input("Start date", 
                                     value=st.session_state.custom_start_date, 
                                     min_value=earliest_date, 
                                     max_value=latest_date)
        end_date   = col2.date_input("End date", 
                                     value=st.session_state.custom_end_date, 
                                     min_value=start_date if start_date else earliest_date, 
                                     max_value=latest_date)
        # Be careful that st.date_input might return a tuple or None
        # TODO: Find better way to ensure this
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

