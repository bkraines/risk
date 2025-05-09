# dates.py

from datetime import timedelta
from typing import Optional, Union



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