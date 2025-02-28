from typing import Union, List
from plotly.graph_objs import Figure

import os

import pandas as pd
import xarray as xr

import plotly.express as px
import plotly.io as pio

from config import IMAGE_DIR


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


def draw_volatility(vol: xr.DataArray, factor_name: str, vol_type: List[int]) -> Figure:
    """
    Draws a line plot of the volatility for a given asset and volatility type.

    Parameters
    ----------
    vol : xr.DataArray
        A DataArray containing the volatility data with dimensions including 'date', 'asset', and 'vol_type'.
    asset : str
        The name of the asset for which the volatility plot is to be drawn.
    vol_type : List[int]
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
    fig = px_line(ds, x='date', y='vol', color='vol_type', title=f'Volatility of {factor_name}')
    return fig


def draw_correlation(corr: xr.DataArray, factor_name: str, factor_name_1: str, corr_type: List[int]) -> Figure:
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
    corr_type : List[int]
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
    fig = px_line(ds, x='date', y='corr', color='corr_type', title=f'Correlation of {factor_name} and {factor_name_1}')
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