# dates.py

from datetime import date, datetime, timedelta
from typing import Optional, Iterable, Union
import pandas as pd
from pandas.tseries.offsets import BDay
import streamlit as st

from risk_lib.config import TRAILING_WINDOWS, MARKET_EVENTS


# def find_prior_month_end(dates: list[pd.Timestamp], current_date: Optional[pd.Timestamp]) -> pd.Timestamp:
#     dates = sorted(pd.to_datetime(dates))
#     if current_date is None:
#         current_date = dates[-1]
#     # current_date = pd.to_datetime(current_date)
#     prior_month_dates = dates[dates < current_date.replace(day=1)]
#     if not prior_month_dates.empty:
#         return prior_month_dates.max()
#     else:
#         return dates.min()  # First date in the sorted list


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



# def select_date_range_bad(
#     date_list: Iterable[date],
#     trailing_windows: Optional[dict[str, int]] = None,
#     named_ranges: Optional[dict[str, tuple[date, date]]] = None
# ) -> tuple[date, date]:
#     """
#     Display a date range selector using Streamlit widgets.

#     Parameters
#     ----------
#     date_list : iterable of date
#         Available dates to select from.
#     trailing_windows : dict of str to int, optional
#         Mapping from label to number of days before the end date. Defaults to TRAILING_WINDOWS.
#     named_ranges : dict of str to tuple(date, date), optional
#         Mapping from label to explicit date ranges. Defaults to MARKET_EVENTS.

#     Returns
#     -------
#     tuple of date
#         The selected start and end dates.
#     """
#     if trailing_windows is None:
#             traling_windows = TRAILING_WINDOWS
#     if named_ranges is None:
#             named_ranges = MARKET_EVENTS
    
#     today = date_list[-1]

#     def get_trailing_date(n: int) -> date:
#         index = max(0, len(date_list) - n - 1)
#         return date_list[index]

#     trailing_options = {
#         label: (get_trailing_date(n), today)
#         for label, n in trailing_windows.items()
#     }

#     to_date_windows = {
#         # TODO: Replace with get_prior_month_end(date_list), get_prior_year_end(date_list)
#         "MTD": get_mtd_range(today),
#         "YTD": get_ytd_range(today),
#     }

#     static_options = {"max": (date_list[0], today), "custom": (None, None)}

#     all_date_options = {
#         **static_options,
#         **to_date_windows,
#         **trailing_options,
#         **named_ranges,
#     }

#     default_option = "max" if "max" in all_date_options else list(all_date_options.keys())[0]
#     label = st.selectbox("Date Range", options=list(all_date_options.keys()), index=list(all_date_options.keys()).index(default_option))

#     start_date, end_date = all_date_options[label]

#     if label == "custom":
#         start_date = st.date_input("Start date", value=today, max_value=today)
#         end_date = st.date_input("End date", value=today, min_value=start_date, max_value=today)

#         if start_date > end_date:
#             st.error("Start date must be before end date.")

#     st.caption(f"{start_date} → {end_date}")
#     return start_date, end_date


def format_date(date_str):
    return pd.to_datetime(date_str).strftime('%m/%d')


def business_days_ago(n=1):
    return (pd.Timestamp.today() - BDay(n)).date()


# TODO: Break out Market Events into separate dropdown? Use an `optgroup` to separate the event types?
# TODO: Allow custom trailing windows?




## REPLACEMENT FROM CLAUDE AI

# DateLike = Union[str, datetime, pd.Timestamp]

# def find_latest_prior_month_date(
#     dates: List[DateLike], 
#     current_date: Optional[DateLike] = None
# ) -> Optional[pd.Timestamp]:
#     """
#     Find the latest date from a prior month in a list of dates.
#     If no dates from a prior month exist, return the earliest date from the current month.
    
#     Parameters
#     ----------
#     dates : List[DateLike]
#         A list of dates in string, datetime, or pandas Timestamp format
#     current_date : Optional[DateLike], default None
#         The reference date used to determine the current month.
#         If None, defaults to the current date (pd.Timestamp.now())
    
#     Returns
#     -------
#     Optional[pd.Timestamp]
#         The latest date from a prior month, or the earliest date from the current month,
#         or None if the input list is empty
    
#     Examples
#     --------
#     >>> dates = ['2023-07-15', '2023-08-01', '2023-08-10', '2023-08-15']
#     >>> find_latest_prior_month_date(dates, '2023-08-20')
#     Timestamp('2023-07-15 00:00:00')
    
#     >>> dates = ['2023-08-01', '2023-08-10', '2023-08-15']
#     >>> find_latest_prior_month_date(dates, '2023-08-20')
#     Timestamp('2023-08-01 00:00:00')
#     """
#     if not dates:
#         return None
    
#     # Convert all dates to pandas Timestamps for consistent handling
#     dates_series = pd.to_datetime(dates)
    
#     # Set current date, defaulting to now if not provided
#     if current_date is None:
#         current_date = pd.Timestamp.now()
#     else:
#         current_date = pd.Timestamp(current_date)
    
#     # Get first day of current month for efficient filtering
#     first_of_month = current_date.replace(day=1)
    
#     # Filter for prior month dates (any date before the first day of current month)
#     prior_month_dates = dates_series[dates_series < first_of_month]
    
#     if not prior_month_dates.empty:
#         # Return the latest date from a prior month
#         return prior_month_dates.max()
#     else:
#         # If no prior month dates, all remaining dates are current month or later
#         # Return the earliest date (which will be the earliest from current month)
#         return dates_series.min()