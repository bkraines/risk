from typing import Optional, Sequence
from datetime import date, datetime

import pandas as pd
from pandas.tseries.offsets import BDay

from risk_util import is_sorted

def format_date(date_str):
    return pd.to_datetime(date_str).strftime(r'%m/%d')


def business_days_ago(n=1, current_date=None):
    if current_date is None:
        current_date = pd.Timestamp.today()
    return (pd.Timestamp.today() - BDay(n)).date()


def latest_business_day(date=None):
    if date is None:
        date = pd.Timestamp.today()
    return (date + BDay(1) - BDay(1)).date()


def get_start_date_n_periods_ago(date_list: Sequence[datetime], n: int) -> datetime:
    assert is_sorted(date_list)
    idx = max(0, len(date_list) - (n + 1))
    return date_list[idx]

# def current_business_day(current_date=None):
#     if current_date is None:
#         current_date = pd.Timestamp.today()
#     return (pd.Timestamp.today().date() + BDay(1) - BDay(1)).date()


def get_mtd_range(today: Optional[date] = None) -> tuple[date, date]:
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
    # TODO: Return the last date of the prior month from a list
    today = today or date.today()
    start = date(today.year, today.month, 1)
    return start, today


def get_ytd_range(today: Optional[date] = None) -> tuple[date, date]:
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


def build_date_options(date_list: Sequence[datetime],
                       rolling_windows: Optional[dict[str, int]],
                       market_events: Optional[dict[str, tuple[datetime, datetime]]]):
    
    assert is_sorted(date_list)
    if rolling_windows is None:
        rolling_windows = {}
    if market_events is None:
        market_events = {}

    earliest_date = min(date_list)
    latest_date = max(date_list)

    static_ranges = {"max": (earliest_date, latest_date),
                     "custom": None,}

    rolling_ranges = {label: (get_start_date_n_periods_ago(date_list, n), latest_date)
                        for label, n in rolling_windows.items()}

    expanding_ranges = {'MTD': get_mtd_range(latest_date),
                        'YTD': get_ytd_range(latest_date)}

    date_options_dict = static_ranges | expanding_ranges | rolling_ranges  | market_events
    return date_options_dict
