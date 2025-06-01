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
# TODO: Include statistics

def get_event_window(
    df: PandasObjectT,
    event_date: str | pd.Timestamp,
    before: int,
    after: int
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
            .assign(day_offset=lambda _df: pd.Series(range(-before, -before + len(_df)), index=_df.index), # pd.Series avoids type error
                    factor_name=factor_name,
                    event_date=event_date,
                    event_name = f"{factor_name};{event_date}",
                    cret = lambda df: df['returns'].cumsum(),
                    cret_centered = lambda df: df['cret'] - df.loc[df['day_offset'] == 0, 'cret'].values[0]
                    )
            .reset_index()
            .set_index(['day_offset', 'factor_name', 'event_date'])
            )
    return pd.concat(_list)


def draw_event_study(returns_df: pd.Series | pd.DataFrame, 
                     event_list: list[tuple[str, pd.Timestamp]], 
                     before: int = 63, 
                     after: int = 252,
                     reverse_y_axis: bool = False) -> Figure:
    """
    Plot an event study figure from returns and event list.

    Parameters
    ----------
    returns_df : pd.Series or pd.DataFrame
        A time series of returns, with `factor_name` as the column if DataFrame.
    event_list : list of tuple(str, pd.Timestamp)
        A list of (factor_name, event_date) pairs specifying the events to plot.
    before : int, default 63
        Number of days before each event to include in the window.
    after : int, default 252
        Number of days after each event to include in the window.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly line chart figure showing centered cumulative returns
        over the event window, colored by event.

    Examples
    --------
    >>> event_list = [('SPY', pd.Timestamp('2024-01-01'))]
    >>> fig = draw_event_study(returns_df, event_list)
    >>> fig.show()
    """
    # TODO: Make prettier hover label
    event_study = run_event_study(returns_df, event_list, before=before, after=after)
    fig = (px.line(event_study.reset_index(), 
                   x='day_offset', 
                   y='cret_centered', 
                   color='event_name', 
                   hover_data={'day_offset': False,
                               'factor_name': False,
                               'event_date': False,
                               'date': r'|%Y-%m-%d',
                               'returns': ':.2f',
                               'cret': ':.2f',
                               'cret_centered': ':.2f',},
                   title='Event Study',
                   template='plotly_white')
        #    .update_traces(
        #        hovertemplate=
        #            'Day Offset: %{x}<br>' +
        #            'cret_centered: %{y:.4f}<br>' +
        #            'Factor: %{customdata[0]}<br>' +
        #            'Raw Return: %{customdata[1]:.4f}<br>' +
        #            'Cumulative Return: %{customdata[2]:.4f}<br>' +
        #            'Event Date: %{customdata[3]}<br>' +
        #            '<extra></extra>'
        #    )
           .update_layout(legend_title_text='Event',
                          hovermode='x unified',
                          xaxis_zeroline=True,
                          xaxis_title_text=None,
                          yaxis_title_text=None,
                       )
           .update_yaxes(autorange='reversed' if reverse_y_axis else True)
    )
    # return px_format(fig)
    return fig

