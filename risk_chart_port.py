from typing import Iterable, Optional
from plotly.graph_objects import Figure

from numpy import sqrt
import pandas as pd
import xarray as xr

import plotly.express as px

from risk_config_port import PORTFOLIOS
from risk_portfolios import portfolio_weights_to_xarray
from risk_chart import px_line

def draw_portfolio_cret_df(returns: pd.DataFrame, unit=10000) -> Figure:
    cumulative_returns = returns[PORTFOLIOS.keys()].div(unit).add(1).cumprod()
    return px.line(cumulative_returns, template='plotly_white')


def draw_portfolio_cret(cret: xr.DataArray, portfolio_names: Optional[Iterable] = None) -> Figure:
    if portfolio_names is not None:
        cret = cret.sel(factor_name=portfolio_names)
    return px_line(cret,
                   x='date',
                   y='cret',
                   color='factor_name',
                   title='Cumulative Return of Portfolios')


def draw_portfolio_weights(portfolio_weights: pd.DataFrame | xr.DataArray) -> Figure:
    # TODO: Use plotly graph objects subplots to create facets independently
    #       allowing for y-coordinate set to vary with facet
    if isinstance(portfolio_weights, pd.DataFrame):
        da = portfolio_weights_to_xarray(portfolio_weights)
    else:
        da = portfolio_weights

    fig = (px.imshow(da.transpose().dropna('date', how='all'),
                     facet_col='portfolio_name',
                     facet_col_wrap=1,
                     height=1200,
                     zmin=-1,
                     zmax=+1,
                     color_continuous_scale='RdBu',
                     template='plotly_white')
           .for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].strip()))
           .for_each_annotation(lambda a: a.update(text='') if a.x == 0 else None)
           .update_layout(coloraxis_showscale=False,
                          xaxis_title=None,
                          yaxis_title=None))
    return fig


def get_portfolio_summary(returns, unit=10000):

    cumulative_returns = returns.div(unit).add(1).cumprod()

    # Define function to compute summary statistics
    def compute_summary_stats(series):
        """
        Computes summary statistics for a given time series of cumulative returns.
        """
        total_period_in_years = (series.index[-1] - series.index[0]).days / 365.25
        cumulative_return = series.iloc[-1] - 1  # cumulative_return over the period
        annualized_return = series.iloc[-1] ** (1 / total_period_in_years) - 1
        # For annualized volatility, use daily returns
        daily_returns = series.pct_change().dropna()
        annualized_volatility = daily_returns.std() * sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility
        return pd.Series({
            'Cumulative Return': cumulative_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio
        })

    # Initialize summary stats DataFrame
    summary_stats_df = pd.DataFrame()

    # Compute summary statistics for portfolios
    for portfolio_name in PORTFOLIOS.keys():
        portfolio_cum_returns = cumulative_returns[portfolio_name]
        stats = compute_summary_stats(portfolio_cum_returns)
        summary_stats_df[portfolio_name] = stats

    # Transpose for better readability
    summary_stats_df = summary_stats_df.T
    return summary_stats_df