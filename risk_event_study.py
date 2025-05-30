from typing import TypeVar
from plotly.graph_objects import Figure

import pandas as pd
import plotly.express as px

from risk_chart import px_format
from risk_util import flatten_multiindex

PandasObjectT = TypeVar("PandasObjectT", pd.Series, pd.DataFrame)

# TODO: Offer a preconstructed list of events
# TODO: Toggle dates on the x-axis
# TODO: Toggle a reverse y-axis
# TODO: Toggle reverse layering on the series
# TODO: Improve tool tips
# TODO: Fix y-axis units
# TODO: Calculate cret inside the event study 
# TODO: Include statistics

def get_event_window(
    df: PandasObjectT,
    event_date: str | pd.Timestamp,
    before: int = 63,
    after: int = 252
) -> PandasObjectT:
    """
    Select a window of rows from a date-indexed DataFrame around an event date.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame with a DateTimeIndex, assumed sorted.
    event_date : str | pd.Timestamp
        The target event date.
    before : int
        Number of rows before the event date to include.
    after : int
        Number of rows after the event date to include.

    Returns:
    -------
    pd.DataFrame
        Sliced DataFrame with [before rows] before and [after rows] after the next available event date.
    """
    
    event_date = pd.Timestamp(event_date)
    # Find next available index position (bfill means ≥ event_date)
    
    event_idx = df.index.get_indexer([event_date], method='bfill')[0]
    if event_idx == -1:
        raise ValueError(f"No date on or after {event_date} found in index!")
    
    start_idx = max(event_idx - before, 0)
    end_idx = event_idx + after + 1  # +1 because iloc is exclusive on end
    return df.iloc[start_idx:end_idx]


def run_event_study(returns_df: pd.Series | pd.DataFrame, 
                    event_list: list[tuple[str, pd.Timestamp]], 
                    before: int = 63, 
                    after: int = 252) -> pd.DataFrame:
    """
    Run an event study given list of events and factors

    Parameters
    ----------
    returns_df : pd.Series or pd.DataFrame
        A time series of returns, with `factor_name` as the column if DataFrame.
    event_list : list of tuple(str, pd.Timestamp)
        A list of (factor_name, event_date) pairs specifying the events to study.
    before : int, default 63
        Number of days before the event to include.
    after : int, default 252
        Number of days after the event to include.

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame (day_offset, factor_name, event_date) with columns:
        - 'returns': original returns in the event window
        - 'cret': cumulative returns over the window
        - 'event_name': a string combining factor_name and event_date

    Notes
    -----
    This function calls `get_event_window()` for each event, builds a DataFrame
    with cumulative returns, and stacks the results into one combined DataFrame.

    Examples
    --------
    >>> event_list = [('SPY', pd.Timestamp('2024-01-01'))]
    >>> run_event_study(returns_df, event_list)
    """
    _list = []
    for factor_name, event_date in event_list:
        _list.append(
            get_event_window(returns_df[factor_name],
                             event_date, 
                             before=before, 
                             after=after)
            .rename('returns')
            .to_frame()
            # .assign(day_offset=range(-before, after + 1)) # THIS MIGHT BE TOO LONG
            .assign(day_offset=lambda _df: pd.Series(range(-before, -before + len(_df)), index=_df.index), # pd.Series avoids type error
                    # day_offset=lambda _df: range(-before, -before + len(_df)), # This gives type error
                    factor_name=factor_name,
                    event_date=event_date,
                    event_name = f"{factor_name};{event_date}",
                    cret = lambda df: df['returns'].cumsum(),
                    cret_centered = lambda df: df['cret'] - df.loc[df['day_offset'] == 0, 'cret'].values[0]
                    # cret=lambda _df: _df.groupby(['factor_name', 'event_date'])['returns'].cumsum(),
                    # cret_centered = lambda df: df['cret'] - (df.loc[df.index == 0].set_index(['factor_name', 'event_date'])['cret'].reindex(df.set_index(['factor_name', 'event_date']).index).values)
                    ).set_index(['day_offset', 'factor_name', 'event_date'])
            )
    return pd.concat(_list)


def draw_event_study(returns_df: pd.Series | pd.DataFrame, 
                     event_list: list[tuple[str, pd.Timestamp]], 
                     before: int = 21, 
                     after: int = 63) -> Figure:
    # TODO: Run cumulative return inside the event study
    event_study = run_event_study(returns_df, event_list, before=before, after=after)
    fig = (px.line(event_study.reset_index(), 
                   x='day_offset', 
                   y='cret_centered', 
                   color='event_name', 
                   hover_data = ['day_offset', 'factor_name', 'returns', 'cret', 'event_date'],
                   title='Event Study',
                   template='plotly_white')
           .update_layout(legend_title_text='Event',
                          xaxis_zeroline=True,
                          ))
    return px_format(fig)


def run_event_study_old(returns_df: pd.Series | pd.DataFrame, 
                    event_list: list[tuple[str, pd.Timestamp]], 
                    before: int = 21, 
                    after: int = 63) -> pd.DataFrame:
    """
    Perform event study using specific factor names tied to events.

    Parameters:
    - returns_df: pd.DataFrame (date index, return columns)
    - event_list: list of (factor_name, event_date)
    - before: days before event
    - after: days after event

    Returns:

    """
    _list = []
    for factor_name, event_date in event_list:
        _list.append(
            get_event_window(returns_df[factor_name],
                             event_date, 
                             before=before, 
                             after=after)
            .rename('returns')
            .to_frame()
            # .assign(day_offset=range(-before, after + 1)) # THIS MIGHT BE TOO LONG
            .assign(day_offset=lambda _df: pd.Series(range(-before, -before + len(_df)), index=_df.index),
                    # day_offset=lambda _df: range(-before, -before + len(_df)), # This gives type error
                    factor_name=factor_name,
                    event_date=event_date,
                    cret=lambda _df: _df.groupby(['factor_name', 'event_date'])['returns'].cumsum(),
                    )
            .reset_index()
            .set_index('day_offset')
        )
    return pd.concat(_list)


def draw_event_study_old(returns_df: pd.Series | pd.DataFrame, 
                     event_list: list[tuple[str, pd.Timestamp]], 
                     before: int = 21, 
                     after: int = 63) -> Figure:
    # TODO: Run cumulative return inside the event study
    event_study = run_event_study(returns_df, event_list, before=before, after=after)
    # event_study.reset_index().to_clipboard()
    df_cum = event_study.reset_index().pivot(index='day_offset', columns=['factor_name', 'event_date'], values='cret') #.cumsum()
    df_cum.columns = flatten_multiindex(df_cum.columns, sep=';')
    df_cum -= df_cum.loc[0]
    fig = (px.line(df_cum, template='plotly_white', title='Event Study')
           .update_layout(legend_title_text='Event',
                          xaxis_zeroline=True,
                          ))
    return px_format(fig)
    # return event_study # df_cum #