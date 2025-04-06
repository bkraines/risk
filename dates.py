# dates.py

from datetime import date, datetime, timedelta
from typing import Optional, Dict, Tuple, Iterable
import pandas as pd
import streamlit as st

from config import TRAILING_WINDOWS, MARKET_EVENTS


def get_mtd_range(today: Optional[date] = None) -> Tuple[date, date]:
    """
    Return the month-to-date range from the first of the current month to today.

    Parameters
    ----------
    today : date, optional
        The reference date. Defaults to today's date.

    Returns
    -------
    tuple of date
        (start_date, end_date)
    """
    today = today or date.today()
    start = date(today.year, today.month, 1)
    return start, today


def get_ytd_range(today: Optional[date] = None) -> Tuple[date, date]:
    """
    Return the year-to-date range from Jan 1 to today.

    Parameters
    ----------
    today : date, optional
        The reference date. Defaults to today's date.

    Returns
    -------
    tuple of date
        (start_date, end_date)
    """
    today = today or date.today()
    start = date(today.year, 1, 1)
    return start, today


def select_date_range(
    date_list: Iterable[date],
    trailing_windows: Optional[Dict[str, int]] = None,
    named_ranges: Optional[Dict[str, Tuple[date, date]]] = None
) -> Tuple[date, date]:
    """
    Display a date range selector using Streamlit widgets.

    Parameters
    ----------
    date_list : iterable of date
        Available dates to select from.
    trailing_windows : dict of str to int, optional
        Mapping from label to number of days before the end date. Defaults to TRAILING_WINDOWS.
    named_ranges : dict of str to tuple(date, date), optional
        Mapping from label to explicit date ranges. Defaults to MARKET_EVENTS.

    Returns
    -------
    tuple of date
        The selected start and end dates.
    """
    trailing_windows = trailing_windows if trailing_windows is not None else TRAILING_WINDOWS
    named_ranges = named_ranges if named_ranges is not None else MARKET_EVENTS

    today = date_list[-1]

    def get_trailing_date(n: int) -> date:
        index = max(0, len(date_list) - n - 1)
        return date_list[index]

    trailing_options = {
        label: (get_trailing_date(n), today)
        for label, n in trailing_windows.items()
    }

    to_date_windows = {
        "MTD": get_mtd_range(today),
        "YTD": get_ytd_range(today),
    }

    static_options = {"max": (date_list[0], today), "custom": (None, None)}

    all_date_options = {
        **static_options,
        **to_date_windows,
        **trailing_options,
        **named_ranges,
    }

    default_option = "max" if "max" in all_date_options else list(all_date_options.keys())[0]
    label = st.selectbox("Date Range", options=list(all_date_options.keys()), index=list(all_date_options.keys()).index(default_option))

    start_date, end_date = all_date_options[label]

    if label == "custom":
        start_date = st.date_input("Start date", value=today, max_value=today)
        end_date = st.date_input("End date", value=today, min_value=start_date, max_value=today)

        if start_date > end_date:
            st.error("Start date must be before end date.")

    st.caption(f"{start_date} → {end_date}")
    return start_date, end_date


def format_date(date_str):
    return pd.to_datetime(date_str).strftime('%m/%d')


# TODO: Break out Market Events into separate dropdown? Use an `optgroup` to separate the event types?
# TODO: Allow custom trailing windows?
