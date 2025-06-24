from typing import Union, Optional, Any
from plotly.graph_objects import Figure
from pandas.io.formats.style import Styler

import os

import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import plotly.io as pio
from plotly.subplots import make_subplots

from risk_config import IMAGE_DIR, VIX_COLORS
from risk_data import get_factor_master
from risk_stats import get_beta_pair, get_zscore

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


def px_format(fig: Figure, x_title: bool = False, y_title: bool = False, annotations: bool = False, hovermode: str = 'x unified') -> Figure:
    if not x_title:
        fig.update_xaxes(title_text=None)
    if not y_title:
        fig.update_yaxes(title_text=None)
    if not annotations:
        fig.for_each_annotation(lambda a: a.update(text=''))

    fig.update_traces(hovertemplate=None)
    fig.update_layout( #legend_title_text=None,
                      hovermode=hovermode,
                      )

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
    plotly.graph_objects._figure.Figure
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
    df = df.loc[:, ~df.columns.duplicated()] # Remove duplicate columns if any
    fig = (px.bar(df, barmode='group', 
                  title=title,
                  template='plotly_white')
            .update_traces(marker_line_width=0,
                           hovertemplate=r'%{y:.2f}')
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
    # TODO: Support different templates
    
    # Get default Plotly color sequence
    default_colors = pio.templates['plotly_white'].layout.colorway
    color1 = default_colors[0]
    color2 = default_colors[1]
    TRANSPARENT_COLOR = 'rgba(0,0,0,0)'

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=series1.index, y=series1.values, name=label1, line_color=color1),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=series2.index, y=series2.values, name=label2, line_color=color2), #, dash='dash')),
        secondary_y=True
    )

    fig.update_layout(
        title=title,
        xaxis_title=None,
        legend_x=0.01, 
        legend_y=0.99,
        legend_bgcolor=TRANSPARENT_COLOR,
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=500,
    )

    # Style y-axes
    fig.update_yaxes(
        secondary_y=False,
        title_text=label1,
        title_font_color=color1,
        tickfont_color=color1,
        showgrid=False,
    )

    fig.update_yaxes(
        secondary_y=True,
        title_text=label2,
        title_font_color=color2,
        tickfont_color=color2,
        showgrid=False,
        
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
    # TODO: Don't hardcode `/100` factor adjustment
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
    # TODO: Don't hardcod `.div(100)`

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
    da = vol.sel(**fig_sel).dropna(dim='date')
    fig = px_line(da, x='date', y='vol', color='vol_type', 
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
    

def format_corr_matrix(corr: pd.DataFrame) -> Styler:
    """
    Format the correlation matrix by adding asset class information and sorting.
    
    Parameters
    ----------
    corr : pd.DataFrame
        The correlation matrix with assets as both rows and columns.
    
    Returns
    -------
    pandas.io.formats.style.Styler
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
) -> Figure:
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
    Figure
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


def transpose_for_plot(da: xr.DataArray, x: str, y: str, t: Optional[str] = None) -> xr.DataArray:
    """
    Return DataArray with dimensions ordered as (y, x[, t]) for plotting.

    Avoids transpose if order is already correct.
    """
    dims = [d for d in (y, x, t) if d is not None]
    if list(da.dims) != dims:
        da = da.transpose(*dims)
    return da


def add_string_coord_for_animation(da: xr.DataArray, 
                                   date_coord: Optional[str]
                                   ) -> tuple[xr.DataArray, Optional[str]]:
    """
    If date_coord is a datetime coordinate, add a string-formatted
    coordinate for animation frames.

    Returns the updated DataArray and the coordinate name to use
    for animation (string coordinate if added, else original).
    """
    if date_coord and isinstance(da.indexes.get(date_coord), pd.DatetimeIndex):
        string_coord = f"{date_coord}_str"
        da = da.assign_coords({
            string_coord: (date_coord, da.coords[date_coord].dt.strftime('%Y-%m-%d').data)
        })
        return da, string_coord
    return da, date_coord


def px_heatmap(da: xr.DataArray, 
               x: str, y: str,
               animation_frame: Optional[str] = None,
               title: Optional[str] = None,
               aspect: str = 'auto',
               color_scale: Optional[str] = 'RdBu_r',
               show_colorbar: bool = False,
               ) -> Figure:
    da = transpose_for_plot(da, x, y, animation_frame)
    if animation_frame is not None:
        da, animation_coord = add_string_coord_for_animation(da, animation_frame)
    else:
        animation_coord = None

    # animation_coord = 'date'
    fig = px.imshow(
        da,
        animation_frame=animation_coord,
        color_continuous_scale=color_scale,
        zmin=-1,
        zmax=+1,
        aspect=aspect,
        origin="lower",
    )

    fig.update_layout(
        coloraxis_showscale=show_colorbar,
        yaxis_autorange="reversed",
        title=title,
        xaxis_title=None,
        yaxis_title=None,
    )


    # if animation_frame == 'date':
    #     fig.for_each_trace(lambda t: t.update(name=pd.to_datetime(t.name).strftime('%Y-%m-%d')))

    return fig
    # return da


def sort_assets(corr: pd.DataFrame, sorting_factor: str, factor_master: Optional[pd.DataFrame]=None) -> pd.Index:
    if factor_master is None:
        factor_master = get_factor_master()
    df = (corr.loc[[sorting_factor]]
          .join(factor_master[['asset_class', 'hyper_factor']])
          .assign(is_theme=lambda df: df['asset_class'] == 'Theme')
          .assign(is_sorting_factor=lambda df: df.index == sorting_factor)
          )
    sorted_index = df.sort_values(by=['is_sorting_factor', 'hyper_factor', 'is_theme', sorting_factor], 
                                  ascending=[False, False, True, False], key=abs).index
    return sorted_index



def draw_corr_matrix(corr: xr.DataArray, 
                     factor_name: str = 'factor_name', 
                     factor_name_1: str = 'factor_name_1', 
                     animation_frame: Optional[str] = None, 
                     corr_type: Optional[int] = None, 
                     height: int = 800, 
                     title: Optional[str] = None,
                     toggle_sort = False) -> Figure:
    # TODO: Height doesn't seem functional
    # TODO: Parameterize animation resampling (monthly?)
    # TODO: Format animation date units
    
    # if animation_frame:
    #     animation_frame = 'date'
    #     da = corr.sel(date=slice('2025', None)) #TODO: Remove this after testing!
    # else:
    #     da = corr.sel(date=corr.date.max()).squeeze()
    da = corr.sel(corr_type=corr_type)

    # da = corr.sel(corr_type=corr_type, date=corr.date.max())
    # da.coords['date'] = da['date'].dt.strftime('%Y-%m-%d')
    
    if title is None:
        title = f'Correlation ({corr_type}-day halflife)'
    fig = (px_heatmap(da, 
                      x=factor_name, 
                      y=factor_name_1, 
                      animation_frame=animation_frame, 
                    #   title=title, 
                      aspect='equal')
           .update_layout(height=height,
                          coloraxis_showscale=False,
                          title=title,
                          xaxis_title=None,
                          yaxis_title=None,
                          xaxis_side='top',
                        #   yaxis_side='right')
                        )
           )
    return fig


def draw_correlation_heatmap(corr: xr.DataArray, fixed_factor: str, 
                             corr_type, 
                             height: int = 1200, title: Optional[str] = None,
                             toggle_sort: bool = False, # NOT WORKING
                             factor_master: Optional[pd.DataFrame] = None) -> Figure:
    # TODO: Resample data weekly or monthly to control which pixels are selected in chart
    # TODO: Fix toggle_sort
    if title is None:
        title = f"Correlation with {fixed_factor} ({corr_type}-day halflife)"

    # if toggle_sort:
    #     corr_latest = corr.sel(date=corr.date.max(), corr_type=corr_type)
    #     factor_list_sorted = sort_assets(corr_latest.to_pandas(), 'VMBS', factor_master)
    #     corr = corr.sel(factor_name=factor_list_sorted)

    da = corr.sel(factor_name_1=fixed_factor, corr_type=corr_type)

    fig = (px_heatmap(da, x='date', y='factor_name', title=title)
           .update_layout(height=height,
                          xaxis_side='top',
                          yaxis_side='right'))
    return fig


def get_color_map(categories: pd.Series) -> dict:
    """
    Generate a color map for unique categories in a pandas Series.

    Parameters
    ----------
    categories : pd.Series
        A pandas Series containing categorical values.

    Returns
    -------
    dict
        A dictionary mapping each unique category to a color string.
    """
    category_list = categories.dropna().unique()
    # palette = pc.qualitative.Plotly
    palette = pc.qualitative.Set3
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(category_list)}
    # rgba_template = 'rgba({}, {}, {}, 0.2)'
    # color_map = {str(cat): rgba_template.format(*pc.hex_to_rgb(palette[i % len(palette)]))
    #                  for i, cat in enumerate(category_list)}

    return color_map


def add_regime_shading(
    fig: Figure, 
    regimes: pd.Series, 
    color_map: Optional[dict] = VIX_COLORS
) -> Figure:
    """
    Adds shaded regions to a Plotly figure based on a regime time series.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly Figure object to be modified.
    regimes : pd.Series
        Series indexed by datetime with categorical regime labels.
    custom_color_map : dict, optional
        Optional mapping of regime name (or np.nan) to RGBA color string.
        If None, mapping is auto-generated by `get_color_map`.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly Figure modified with background shading.

    Notes
    -----
    - Shading is only applied for non-NaN regime blocks.

    """
    # TODO: Add build_regime_vrects function, which add_regime_shading calls if rectanges aren't cached
    # TODO: Shading ends at the final date marker, which may not be chart's right edge
    #       (e.g. shading in `px.bar` charts with `barmode="group"` stops at the center of the final bar group)
    if regimes.empty:
        return fig

    if color_map is None:
        color_map = get_color_map(regimes)
    default_color = f'rgba(128, 128, 128, 0.2)'
    regime_blocks = (regimes
                     .ffill()
                     .to_frame(name='regime')
                     .assign(group=(lambda df: (df['regime'] != df['regime'].shift()).cumsum()))
                     )
    date_list = regimes.index
    vrects = []
    for (group, regime), block in regime_blocks.groupby(['group', 'regime']):
        if pd.isna(regime):
            continue
        start_date = block.index.min()
        end_date = block.index.max()
        # Draw the rectangle to the next valid date (don't end on a Friday, stretch to Monday)
        # TODO: If it's the last block, ensure coverage to chart edge
        # TODO: Try speeding up with `searchsorted`
        end_date_next = (date_list[date_list > end_date].min()
                         if date_list.max() > end_date
                         else date_list.max()) # + pd.DateOffset(years=1)) # doesn't work
        color = color_map.get(regime, default_color)
        vrects.append(dict(
            xref='x',
            x0=start_date,
            x1=end_date_next,
            yref='paper',
            y0=0,
            y1=1,
            fillcolor=color,
            # opacity=1.0.
            layer='below',
            line_width=0))
        # fig.add_vrect(
        #     x0=start_date,
        #     x1=end_date_next,
        #     fillcolor=color,
        #     # opacity=1.0.
        #     layer='below',
        #     line_width=0
        # )
    fig.update_layout(shapes=fig.layout.shapes + tuple(vrects))  
    return fig