from datetime import date, datetime
from typing import Iterable, Optional
import streamlit as st

import os
from risk_lib.data import get_factor_data
from risk_lib.util import check_memory_usage, get_mtd_range, get_ytd_range, summarize_memory_usage, get_directory_last_updated_time
from risk_lib.config import CACHE_TARGET, MARKET_EVENTS, TRAILING_WINDOWS, CACHE_DIR, CACHE_FILENAME


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
    trailing_windows: Optional[dict[str, int]] = TRAILING_WINDOWS,
    market_events: Optional[dict[str, tuple[datetime, datetime]]] = MARKET_EVENTS,
    # to_date_windows: Optional[dict[str, Callable[[datetime], tuple[datetime, datetime]]]] = TO_DATE_WINDOWS,
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
    # TODO: Rename trailing_windows[ranges] to rolling_windows[ranges]
    #       and to_date_windows[ranges] to expanding[ranges]
    if trailing_windows is None:
        trailing_windows = {}
    if market_events is None:
        market_events = {}

    date_list = sorted(date_list)
    if not date_list:
        raise ValueError(f"List of dates must be nonempty")
    # if not date_list:
    #     st.error("date_list must contain at least one date.")
    #     return None, None

    default_start = date_list[0]
    latest_date = date_list[-1]

    def get_start_date_n_periods_ago(n: int) -> datetime:
        idx = max(0, len(date_list) - (n + 1))
        return date_list[idx]

    static_ranges = {"max":   (default_start, latest_date),
                     "custom": None,}

    trailing_ranges = {label: (get_start_date_n_periods_ago(n), latest_date)
                       for label, n in trailing_windows.items()}

    to_date_ranges = {'MTD': get_mtd_range(latest_date),
                      'YTD': get_ytd_range(latest_date)}

    # to_date_windows = {"MTD": get_mtd_range,
    #                    "YTD": get_ytd_range}

    # to_date_ranges = {label: func(latest_date)
    #                   for label, func in to_date_windows.items()}

    all_date_options = static_ranges | to_date_ranges | trailing_ranges  | market_events
    options = list(all_date_options.keys())

    if default_option and (default_option in options):
        initial_selection = default_option
    elif "max" in options:
        initial_selection = "max"
    else:
        initial_selection = options[0]

    selected_range = st.selectbox("Date Range", options, index=options.index(initial_selection))

    if selected_range != "custom":
        start_date, end_date = all_date_options[selected_range]
        st.write(f"{start_date.strftime(r'%Y-%m-%d')} to {end_date.strftime(r'%Y-%m-%d')}") # TODO: 
    else:
        start_date = st.date_input("Start date", default_start, min_value=default_start, max_value=latest_date)
        end_date = st.date_input("End date", latest_date, min_value=default_start, max_value=latest_date)

        if not isinstance(start_date, date):
            raise ValueError(f"Expected a single date, but got {start_date} of type {type(start_date)}")
        if not isinstance(end_date, date):
            raise ValueError(f"Expected a single date, but got {end_date} of type {type(end_date)}")

        # start_date = start_date[0] if isinstance(start_date, tuple) else start_date
        # end_date = end_date[0] if isinstance(end_date, tuple) else end_date

        if start_date > end_date:
            st.error("Start date must be before end date.")
        st.write(f"{start_date.strftime(r'%Y-%m-%d')} to {end_date.strftime(r'%Y-%m-%d')}")

    # TODO: Ensure start_date and end_date are of comparable types then print output in one place:
    # if start_date > end_date:
    #     st.error("Start date must be before end date.")     
    # else:
    #     st.write(f"{start_date.strftime(r'%Y-%m-%d')} to {end_date.strftime(r'%Y-%m-%d')}")        


    # TODO: Break out Market Events into separate dropdown? Use an `optgroup` to separate the event types?
    # TODO: Allow custom trailing windows?
    return start_date, end_date