from typing import Union, Optional, Any
from plotly.graph_objs import Figure

import os

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from risk_lib.config import IMAGE_DIR
from risk_lib.stats import get_beta_pair, get_zscore

# TODO: Share chart dimensions and template by extracting `fig_format_default` dict from `px_line`
# TODO: Pull ploty template into a constant
# TODO: Pull default width and height into a constant
PLOTLY_TEMPLATE = 'plotly_white'


def px_write(fig, filename, directory=IMAGE_DIR):
    
    def get_extension(filename):
        return os.path.splitext(filename)[1][1:]

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory, filename)
    if get_extension(filename) == 'html':
        fig.write_html(file_path)
    else:
        fig.write_image(file_path)


def px_format(fig: Figure, x_title: bool = False, y_title: bool = False, annotations: bool = False) -> Figure:
    if not x_title:
        fig.update_xaxes(title_text=None)
    if not y_title:
        fig.update_yaxes(title_text=None)
    if not annotations:
        fig.for_each_annotation(lambda a: a.update(text=''))
        
    return fig


def px_line(da: xr.DataArray, x: str, y: str, color: Union[str, None] = None, title: Union[str, None] = None, 
            x_title: bool = False, y_title: bool = False, fig_format: Union[dict, None] = None) -> Figure:
    # TODO: This should take a Series instead of a DataArray
    fig_format_default = {'template': 'plotly_white', 'height': 500, 'width': 1000}
    fig_format = {**fig_format_default, **(fig_format or {})}
    df = da.to_series().reset_index()
    fig = px.line(df, x=x, y=y, color=color, title=title, **fig_format)
    fig = px_format(fig, x_title=x_title, y_title=y_title)
    
    return fig


def px_scatter(df, x, y, color_map_override=None, **kwargs):
    """
    Create a scatter plot using Plotly with customized color mapping and formatting.
    Parameters
    ----------
    df : pandas.DataFrame
        The data frame containing the data to be plotted.
    x : str
        The column name to be used for the x-axis.
    y : str
        The column name to be used for the y-axis.
    color_map_override : dict, optional
        A dictionary to override the default color sequence for specific categories.
    **kwargs : dict
        Additional keyword arguments to be passed to the plotly.express.scatter function. Typical arguments include:
        `text`, `color`, `size`, and `symbol`, which contain column names for those attributes.
    Returns
    -------
    plotly.graph_objs._figure.Figure
        The generated scatter plot figure.
    Notes
    -----
    The function uses the 'plotly_white' template by default and sets the figure height and width to 750.
    If 'color' is provided in kwargs and 'color_map_override' is not None, a custom color mapping is applied.
    The text font color for each trace is updated based on the provided color map override.
    """
    args_format = {'template': 'plotly_white', 'height': 750, 'width': 750}
    
    color = kwargs.get('color')
    # TODO: Refactor to get_color_map() function
    if (color is None) or (color_map_override is None):
        color_discrete_map = None
        color_map_override = {}
    else:
        color_keys = df[color].unique()
        color_sequence = pio.templates[args_format['template']]['layout']['colorway']
        color_dict = {a: b for a, b in zip(color_keys, color_sequence)}
        color_discrete_map = {**color_dict, **color_map_override}

    fig = (px.scatter(df, x=x, y=y, **kwargs, #text=text, color=color, size=size, symbol=symbol, 
                      color_discrete_map=color_discrete_map,
                      size_max=20,
                      **args_format)
           .update_traces(textposition='middle right', textfont_color='lightgray')
           .update_layout(legend_title_text=None)
           )

    def get_trace_color(trace, legendgroup):
        return trace.marker.color if trace.legendgroup in legendgroup else 'lightgray'

    asset_class_list = color_map_override.keys()
    fig.for_each_trace(lambda t: t.update(textfont_color = get_trace_color(t, asset_class_list)))

    return fig


def px_bar(da: xr.DataArray, title: Optional[str] = None) -> Figure:
    df = da.to_pandas()
    fig = (px.bar(df, barmode='group', 
                  title=title,
                  template='plotly_white')
            .update_traces(marker_line_width=0,
                           hovertemplate=None)
            .update_layout(xaxis_title=None,
                           yaxis_title=None,
                           legend_title_text=None,
                        #    legend=dict(x=0.01, y=0.99),
                           hovermode='x unified')
            )
    return px_format(fig)


def plot_dual_axis(series1: pd.Series, series2: pd.Series, 
                   label1='Series 1', label2='Series 2', 
                   title='Dual Axis Plot'):
    """
    Plot two time series on the same x-axis with separate y-axes,
    and match y-axis colors to line colors (automatically selected).

    Args:
        series1 (pd.Series): Time series for the primary y-axis.
        series2 (pd.Series): Time series for the secondary y-axis.
        label1 (str): Label for the primary y-axis series.
        label2 (str): Label for the secondary y-axis series.
        title (str): Plot title.
    """
    # TODO: Clean up this ChatGPT output
    # TODO: Draw primary line with `px_line` to inherit formatting
    
    # Get default Plotly color sequence
    default_colors = pio.templates['plotly_white'].layout.colorway
    color1 = default_colors[0]
    color2 = default_colors[1]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=series1.index, y=series1.values, name=label1, line=dict(color=color1)),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=series2.index, y=series2.values, name=label2, line=dict(color=color2)), #, dash='dash')),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        # xaxis_title='Date',
        xaxis_title=None,
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=500,
    )

    # Style y-axes
    fig.update_yaxes(
        title_text=label1,
        title_font=dict(color=color1),
        tickfont=dict(color=color1),
        showgrid=False,
        secondary_y=False
    )

    fig.update_yaxes(
        title_text=label2,
        title_font=dict(color=color2),
        tickfont=dict(color=color2),
        showgrid=False,
        secondary_y=True
    )

    return fig


def draw_cumulative_return(da: xr.DataArray, factor_name: str, factor_name_1: str) -> Figure:
    # TODO: Scale secondary factor to same units as primary factor given a start date
    #       Alternatively, start both at 100
    df1 = da.sel(factor_name=factor_name).to_series()
    df2 = da.sel(factor_name=factor_name_1).to_series()
    fig = plot_dual_axis(df1, df2, label1=factor_name, label2=factor_name_1, title=f'{factor_name} vs {factor_name_1}')
    return fig


def draw_returns(ret: xr.DataArray, factor_name: str, factor_name_1: str) -> Figure:
    da = ret.sel(factor_name=[factor_name, factor_name_1]) / 100
    return px_bar(da, title=f'Daily Returns of {factor_name} and {factor_name_1} (%)')


def draw_zscore(ret: xr.DataArray, vol: xr.DataArray, factor_name: str, factor_name_1: str, vol_type) -> Figure:
    ret = ret.sel(factor_name=[factor_name, factor_name_1])
    vol = vol.sel(factor_name=[factor_name, factor_name_1], vol_type=vol_type)
    zscore = get_zscore(ret, vol, 1)
    return px_bar(zscore, title=f'Daily Returns of {factor_name} and {factor_name_1} (std, {vol_type}-day vol)')


def draw_returns_old(ret: xr.DataArray, factor_name: str, factor_name_1: str) -> Figure:
    # TODO: Extract `px.bar(...)` to function named `px_bar()`
    # TODO: Add zscore chart (Display % and std units? Add toggle?)

    df = ret.to_pandas()[[factor_name, factor_name_1]].div(100)
    fig = (px.bar(df, barmode='group', 
                   title=f'Daily Returns of {factor_name} and {factor_name_1} (%)',
                   template='plotly_white')
            .update_traces(marker_line_width=0,
                           hovertemplate=None)
            .update_layout(xaxis_title=None,
                           yaxis_title=None,
                           legend_title_text=None,
                        #    legend=dict(x=0.01, y=0.99),
                           hovermode='x unified')
            )
    return px_format(fig)


def draw_volatility(vol: xr.DataArray, factor_name: str, vol_type: list[int]) -> Figure:
    """
    Draws a line plot of the volatility for a given asset and volatility type.

    Parameters
    ----------
    vol : xr.DataArray
        A DataArray containing the volatility data with dimensions including 'date', 'asset', and 'vol_type'.
    asset : str
        The name of the asset for which the volatility plot is to be drawn.
    vol_type : list[int]
        A list of integers representing the types of volatility to be plotted.

    Returns
    -------
    fig : Figure
        A plotly Figure object representing the volatility line plot.

    Notes
    -----
    The function selects the data for the specified asset and volatility types, drops any NaN values along the 'date' dimension, 
    and then creates a line plot using plotly express.
    """
    fig_sel = {'factor_name': factor_name,
               'vol_type': vol_type,
               }
    ds = vol.sel(**fig_sel).dropna(dim='date')
    fig = px_line(ds, x='date', y='vol', color='vol_type', 
                  title=f'Volatility of {factor_name}')
    return fig


def draw_correlation(corr: xr.DataArray, factor_name: str, factor_name_1: str, corr_type: list[int]) -> Figure:
    """
    Draws a correlation plot between two assets over time.

    Parameters
    ----------
    corr : xr.DataArray
        A DataArray containing correlation data with dimensions including 'date', 'asset', 'asset_1', and 'corr_type'.
    asset : str
        The name of the first asset.
    asset_1 : str
        The name of the second asset.
    corr_type : list[int]
        A list of correlation types to be plotted.

    Returns
    -------
    fig : Figure
        A plotly Figure object representing the correlation plot.

    Examples
    --------
    >>> corr = xr.DataArray(...)
    >>> fig = draw_correlation(corr, 'Asset_A', 'Asset_B', [1, 2, 3])
    >>> fig.show()
    """
    fig_sel = {'factor_name': factor_name,
               'factor_name_1': factor_name_1,
               'corr_type': corr_type,
               }
    ds = corr.sel(**fig_sel).dropna(dim='date')
    fig = px_line(ds, x='date', y='corr', color='corr_type', 
                  title=f'Correlation of {factor_name} and {factor_name_1}')
    return fig
    

def format_corr_matrix(corr: pd.DataFrame): # -> pd.io.formats.style.Styler:
    """
    Format the correlation matrix by adding asset class information and sorting.
    
    Parameters
    ----------
    corr : pd.DataFrame
        The correlation matrix with assets as both rows and columns.
    
    Returns
    -------
    pd.io.formats.style.Styler
        The formatted correlation matrix as a pandas Styler object.
    """
    # Sort the correlation matrix by asset class
    corr = corr
    
    # Style the correlation matrix
    styled_corr = corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format(precision=2)
    
    return styled_corr


def draw_beta(factor_data: xr.Dataset, factor_name: str, factor_name_1: str) -> Figure:    
    # cov_type = zip(vol_type, corr_type)
    beta = get_beta_pair(factor_data.vol, factor_data.corr, factor_name, factor_name_1)
    ds = beta.dropna(dim='date')
    fig = (px_line(ds, x='date', y='beta', color='cov_type', 
                   title=f'Beta of {factor_name_1} to {factor_name}')
        #    .update_yaxes(type="log")
           )
    
    return fig


def draw_volatility_ratio(vol: xr.DataArray, factor_name: str, factor_name_1: str, vol_type: list[int]) -> Figure:
    
    vol_0 = vol.sel(factor_name=factor_name, vol_type=vol_type)
    vol_1 = vol.sel(factor_name=factor_name_1, vol_type=vol_type)
    vol_ratio = vol_1 / vol_0
    
    ds = vol_ratio.dropna(dim='date')
    fig = px_line(ds, x='date', y='vol', color='vol_type', 
                  title=f'Volatility Ratio of {factor_name_1} to {factor_name}')
    return fig


def draw_distance_from_ma(dist_ma: xr.DataArray, factor_name: str, factor_name_1: str, window: int = 200) -> Figure:
    # _cret = cret.sel(factor_name=[factor_name, factor_name_1])
    # dist_ma = distance_from_moving_average(_cret, window)
    _dist_ma = dist_ma.sel(factor_name=[factor_name, factor_name_1], ma_type=window)
    fig = px_line(_dist_ma, x='date', y='dist_ma', color='factor_name', 
                  title=f'Distance from {window}-day Moving Average (%)')
    return fig


def draw_days_from_ma(days_ma: xr.DataArray, factor_name: str, factor_name_1: str, window: int = 200, vol_type: int = 63) -> Figure:
    # _cret = cret.sel(factor_name=[factor_name, factor_name_1])
    # dist_ma = distance_from_moving_average(_cret, window)
    _days_ma = days_ma.sel(factor_name=[factor_name, factor_name_1], ma_type=window, vol_type=vol_type)
    fig = px_line(_days_ma, x='date', y='days_ma', color='factor_name', 
                  title=f'Distance from {window}-day Moving Average (std, {vol_type}-day halflife)')
    return fig


def add_45_degree_line(fig: Figure, 
                       min_val: float, 
                       max_val: float, 
                       line_color: str = "rgba(204, 204, 204, 1)", 
                       line_width: int = 1) -> Figure:
    line = go.Scatter(x=[min_val, max_val],
                      y=[min_val, max_val],
                      mode="lines",
                      line_color=line_color,
                      line_width=line_width,
                      showlegend=False,
                      hoverinfo="skip")
    return fig.add_trace(line)

def get_dist_name(dist) -> str:
    # Get the name of a scipy.stats distribution
    return type(dist).__name__.replace("_gen", "").capitalize()

def plot_qq(
    df: pd.DataFrame,
    dist: stats.rv_continuous = stats.norm,
    title: str = "QQ Plot",
    width: int = 600,
    height: int = 600,
) -> go.Figure:
    """
    Generate a QQ-plot comparing columns of a DataFrame to a theoretical distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where each column is a variable to be plotted.
    dist : stats.rv_continuous, default stats.norm
        A scipy.stats continuous distribution object.
    title : str, default "QQ Plot"
        Title of the plot.
    width : int, default 600
        Width of the figure in pixels.
    height : int, default 600
        Height of the figure in pixels.

    Returns
    -------
    go.Figure
        A Plotly figure object showing the QQ plot.
    """
    # TODO: Only drop from the top of the DataFrame
    df = df.dropna()
    n = len(df)
    if n == 0:
        raise ValueError("Input DataFrame is empty after dropping missing values.")

    probs = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    theoretical_quantiles = pd.Series(dist.ppf(probs), 
                                      index=df.index)

    fig = go.Figure()
    for col in df.columns:
        sample_quantiles = df[col].sort_values()
        fig.add_trace(
            go.Scatter(
                name=col,
                x=theoretical_quantiles,
                y=sample_quantiles.values,
                text=sample_quantiles.index.strftime(r"%Y-%m-%d"),
                mode="markers",
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    "%{text}<br>"
                    "Theoretical: %{x:.2f}<br>"
                    "Sample: %{y:.2f}<extra></extra>"
                ),
            )
        )
    
    #TODO: Include theoretical quantiles in extrema for 45-degree line?
    # all_vals = np.concatenate([theoretical_quantiles, df.values.ravel()])
    # min_val = np.nanmin(all_vals)
    # max_val = np.nanmax(all_vals)
    
    fig = (add_45_degree_line(fig, 
                              df.min().min(),  
                              df.max().max())
           .update_layout(title=title,
                          xaxis_title=f"Theoretical Quantiles ({get_dist_name(dist)})",
                          yaxis_title="Sample Quantiles",
                          template="plotly_white",
                          width=width,
                          height=height)
           .update_yaxes(scaleanchor="x", 
                         scaleratio=1)
           )

    return fig


def draw_zscore_qq(ret: xr.DataArray, vol: xr.DataArray, factor_vol_pairs: list[tuple[str, Any]]) -> Figure:
    """
    Draw a QQ plot comparing standardized returns (z-scores) across multiple factors and vol models.

    Parameters
    ----------
    ret : xr.DataArray
        Array of returns with a 'factor_name' coordinate.
    vol : xr.DataArray
        Array of volatility values with 'factor_name' and 'vol_type' coordinates.
    factor_vol_pairs : list of tuple of str
        List of (factor_name, vol_type) pairs for which to display z-scores.

    Returns
    -------
    Figure
        Plotly figure containing the QQ plot with all specified factors.
    """
    # TODO: Alternatively accept array of z-scores directly
    zscore_dict = {f"{f} ({v})": (get_zscore(ret.sel(factor_name=f),
                                             vol.sel(factor_name=f, vol_type=v))
                                  .to_series())
                   for f, v in factor_vol_pairs}
    df = pd.concat(zscore_dict, axis=1).dropna()
    fig = plot_qq(df, title="QQ Plot")
    return fig

def draw_zscore_qq_single(ret: xr.DataArray, vol: xr.DataArray, factor_name: str, vol_type: Any) -> Figure:
    return draw_zscore_qq(ret, vol, [(factor_name, vol_type)])