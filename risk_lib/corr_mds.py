from typing import Union, Literal, Optional

from numpy import sqrt, cos, sin, arctan2, array #, identity
# import numpy as np
import pandas as pd
import xarray as xr
xr.set_options(keep_attrs=True,
               display_expand_data=False)
from sklearn.manifold import MDS

import plotly.express as px
import plotly.io as pio
# pio.renderers.default='png'
import plotly.graph_objects as go
# from plotly.graph_objs import Figure

# from risk_lib.util import to_pandas_strict

def prepare_correlation(corr: xr.DataArray, transformation=None, start_date=None) -> pd.DataFrame:
    factor_master = pd.DataFrame(corr.asset.attrs).T
    return (mds_ts_df(corr, 
                      transformation=transformation, 
                      start_date=start_date)
            .reset_index()
            .join(factor_master, on='asset')
            .assign(date = lambda df: df['date'].astype(str)))


def transform_coordinates(coordinates: pd.DataFrame, transformation_type=None, factor: str = 'SPY', 
                          factor_list = None, coordinates_initial: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    
    def get_rotation_matrix(theta):
        return array([[cos(theta), -sin(theta)],
                      [sin(theta), cos(theta)]])
        
    def rotate(coordinates, theta):
        rotation_matrix = get_rotation_matrix(theta)
        transformed_coordinates = (rotation_matrix @ coordinates.T).T
        transformed_coordinates.columns = coordinates.columns
        return transformed_coordinates #.rename(columns=dict(enumerate(df_t.columns)))
    
    if transformation_type is None:
        return coordinates
    if coordinates_initial is None:
        coordinates_initial = coordinates
    
    match transformation_type:
        case 'rotate':
            v = coordinates.loc[factor]
            x, y = v
            theta = arctan2(y, x)
            return coordinates.pipe(rotate, -theta)
    
        case 'rotate_initial':
            v = coordinates_initial.loc[factor]
            x, y = v
            theta = arctan2(y, x)
            return coordinates.pipe(rotate, -theta)
    
        case 'normalize':
            v = coordinates.loc[factor]
            x, y = v
            theta = arctan2(y, x)
            length = sqrt(sum(v**2))
            return coordinates.pipe(rotate, -theta).div(length)
    
        case 'rotate_list':
            thetas = {}
            if factor_list is None:
                factor_list = ['SPY', 'TLT']
            
            for factor in factor_list:
                v = coordinates.loc[factor]
                x, y = v
                thetas[factor] = arctan2(y, x)
            theta_avg = sum(thetas.values()) / len(thetas)
            return coordinates.pipe(rotate, -theta_avg)

        case _:
            return coordinates
        


def multidimensional_scaling(correlation_matrix: pd.DataFrame, init=None, random_state=42, n_init=4) -> pd.DataFrame:
    # TODO: Will this accept dataarray?
    # TODO: Does it preserve distance? YES, see mds_test.ipynb
        # If not, enforce distance is economic.
        # Should match sqrt(1-r^2). 
        # Rotate by -avg(theta) of hyperfactors
    
    """
    Perform multidimensional scaling on a correlation matrix.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        The input correlation matrix as a pandas DataFrame.
    init : np.ndarray, optional
        Initial positions of the points in the embedding space. If None, random initialization is used.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the MDS results with dimensions 'dim1' and 'dim2'.
    """
    
    dissimilarity_matrix = sqrt(1 - correlation_matrix**2)

    # Pass n_init explicitly to suppress warning when init is not None:
    # if n_init is None:
    #     n_init =
    # n_init = 100  if init is not None else 1 
    embedding = MDS(dissimilarity='precomputed', random_state=random_state, n_init=n_init)
    coordinates = embedding.fit_transform(dissimilarity_matrix, init=init)
    
    return pd.DataFrame(coordinates, 
                        index=dissimilarity_matrix.index, 
                        columns=pd.Index(['dim1', 'dim2'], name='dimension'))


def mds_ts_df(corr: xr.DataArray, start_date=None, transformation=None, factor='SPY', **kwargs) -> pd.DataFrame:
    # TODO: Factor out corr_type

    dates = corr.sel(date=slice(start_date, None)).date.values
    # coordinates = None
    transformed = None
    coordinates = None

    if transformation == 'rotate_initial':
        date = dates.max()
        # Choose halflife close to 63 days:
        df = corr.sel(date=date, corr_type=63, method='nearest').to_pandas() # _strict() # * 0 + np.identity(len(corr.asset))
        coordinates = multidimensional_scaling(df, init=transformed) #init=coordinates) 
        transformed_initial = transform_coordinates(coordinates, 'rotate', factor='SPY')
        transformation = None
    else:
        transformed_initial = None
  
    mds_dict = {}
    for date in dates:
        # Choose halflife close to 63 days
        df = corr.sel(date=date, corr_type=63, method='nearest').to_pandas() #_strict() # * 0 + np.identity(len(corr.asset))
        coordinates = multidimensional_scaling(df, init=transformed_initial, n_init=1) #init=transformed) 
        transformed = transform_coordinates(coordinates, transformation, factor=factor, 
                                            factor_list=None, coordinates_initial=transformed_initial)
        mds_dict[date] = transformed
    return (pd.concat(mds_dict)
            .rename_axis(index=['date', df.index.name])
            )


def draw_mds_ts(df: pd.DataFrame, tick_range: Union[None, float, Literal['auto']] = 'auto') -> go.Figure:
    """
    Draws a scatter plot of MDS (Multidimensional Scaling) time series data using Plotly.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the MDS data with columns 'dim1', 'dim2', 'asset', 'asset_class', and, optionally, 'date'.
    tick_range : Union[None, float, Literal['auto']], optional
        Range for the x and y axis ticks. If 'auto', the range is determined automatically based on the data.
        If None, the range is not set. Default is 'auto'.
    Returns
    -------
    plotly.graph_objs._figure.Figure
        A Plotly Figure object representing a scatter plot of the MDS time series data, animated with a range slider if `date` is present.
    Examples
    --------
    >>> import pandas as pd
    >>> from plotly.graph_objs import Figure
    >>> data = {
    ...     'dim1': [0.1, 0.2, 0.3],
    ...     'dim2': [0.4, 0.5, 0.6],
    ...     'asset': ['Asset1', 'Asset2', 'Asset3'],
    ...     'asset_class': ['Class1', 'Class2', 'Class3'],
    ...     'date': ['2021-01-01', '2021-01-02', '2021-01-03']
    ... }
    >>> df = pd.DataFrame(data)
    >>> fig = draw_mds_ts(df)
    >>> fig.show()
    """
    
    args_animation = {'animation_frame': 'date', 'animation_group': 'factor_name'}  if 'date' in df.columns else {}
    args_format    = {'template': 'plotly_white', 'height': 750, 'width': 750}
    args_size      = {'size': 'size', 'size_max': 15} if 'size' in df.columns else {}
    # args_size      = {'size': 'marker_size'} if 'marker_size' in df.columns else {}
    args_symbol    = {'symbol': 'marker_symbol'} if 'marker_symbol' in df.columns else {}
    # args_textcolor = ['black' if condition else 'lightgray' for condition in (df['asset']=='SPY')]
    # args_textcolor = ['black' if asset == 'SPY' else 'lightgray' for asset in df['asset']]
    df['textcolor'] = ['black' if asset == 'SPY' else 'lightgray' for asset in df['factor_name']]
    
    
    asset_class_list = df.asset_class.unique()
    color_sequence = pio.templates['plotly_white']['layout']['colorway']
    color_dict = {a: b for a, b in zip(asset_class_list, color_sequence)}
    color_dict['Portfolio'] = 'black'
    color_dict['Theme'] = 'red'
    
    fig = (px.scatter(df, 
                      x='dim1', y='dim2', text='factor_name', color='asset_class', 
                      color_discrete_map=color_dict,
                      **args_size,
                      **args_symbol,
                      **args_animation,
                      **args_format)
           .update_traces(textposition='middle right', 
                        #   textfont_color='lightgray',
                        #   textfont_color=df['textcolor'], #args_textcolor,
                          )
           .update_layout(xaxis_title=None,
                          yaxis_title=None,
                          xaxis_scaleanchor="y", 
                          xaxis_scaleratio=1,
                          yaxis_tickvals = [-0.5, 0, 0.5],
                          legend_title_text=None)
           )
    
    # fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color)) #, textposition='top center'))
    
    # fig.update_layout(yaxis_tickvals = fig)
    
    # Refactor this into separate function
    if tick_range is not None:
        if tick_range == 'auto':
            tick_range = df[['dim1', 'dim2']].abs().max().max()
        (fig.update_xaxes(range=(-tick_range, tick_range))
            .update_yaxes(range=(-tick_range, tick_range)))
    
    return fig


def add_whiskers(fig, df, t0, t1):
    df0 = df.set_index('date').xs(t0)
    df1 = df.set_index('date').xs(t1)
    for i in range(len(df0)):
        fig.add_trace(go.Scatter(x=[df0['dim1'].iloc[i], 
                                    df1['dim1'].iloc[i]], 
                                 y=[df0['dim2'].iloc[i], 
                                    df1['dim2'].iloc[i]], 
                                 mode='lines', 
                                 line_color='lightgray', 
                                 line_width=1, 
                                 showlegend=False))
        
        # Move whiskers underneath existing traces
        fig.data = tuple([fig.data[-1]] + list(fig.data[:-1]))
    return fig


def get_marker_size(ds):
    # date = '2024-11-01'
    date_latest = ds.date.max().values
    # Choose as close to 21 days as possible
    vol_short = ds['vol'].sel(date=date_latest, vol_type=21, method='nearest')
    vol_long  = ds['vol'].sel(date=date_latest, vol_type=126)
    vol_ratio = vol_short / vol_long
    vol_ratio.to_pandas() #.rename('vol_ratio') # .sort_values(ascending=False)

    factor_master = pd.DataFrame(ds.factor_name.attrs).T

    df = factor_master[['hyper_factor']].join(vol_ratio.to_pandas().rename('vol_ratio'))
    # df['vol_ratio'].sort_values(ascending=False).mul(5).clip(upper=10)

    df['marker_size'] = (df['vol_ratio'].add(1).mul(5).clip(upper=10) #, lower=1)
                         .where(df['hyper_factor'] != 1, 15)
                         )
    date_prior = ds.date.sel(date=slice(None, date_latest)).isel(date=-21).values
    cret_t1 = ds['cret'].sel(date=date_latest)
    cret_t0 = ds['cret'].sel(date=date_prior, method='nearest')
    ret = ((cret_t1/cret_t0)-1).to_pandas()
    
    df['marker_symbol'] = ret.map(lambda x: 'circle' if x > 0 else 'triangle-up')

    return df['marker_size'] #, 'marker_symbol']]


def run_mds(ds, transformation, dates, start_date, tick_range, animate=False, drop_composites=True, drop_trump=False, **kwargs):
    # TODO: Pass in full dataset to extract corr, factor_master, and vol (for sizing)
    
    # TODO: Pass in a list of dates or take all dates from the dataarray
    # TODO: Make clear the ordering of dates (use sorted function)
    # t0, t1, t2 = dates
    # (t0, t1, t2) = factor_data2.date.values[[-1, -21-1, -63-1]]

    
    transformation_type = {None:             'No rotation', 
                           'rotate':         'Rotate SPY to x-axis each day', 
                           'normalize':      'SPY transformed to (1, 0)',
                           'rotate_initial': '' #'Rotate SPY to x-axis today'
                           }

    factor_master = pd.DataFrame(ds.factor_name.attrs).T
    
    marker_size = get_marker_size(ds) #.rename('size')
    
    mds_ts = (mds_ts_df(ds.corr, transformation=transformation, start_date=start_date, **kwargs)
                .reset_index()
                .join(factor_master, on='factor_name')
                .assign(date = lambda df: df['date'].astype(str))
                # .assign(size = lambda df: df['hyper_factor'].mul(1).add(.5).astype('float'))
                # .assign(size = lambda df: df['hyper_factor'].apply(lambda x: 10 if x == 1 else 3).astype('float'))
                # .assign(size = lamdba df: marker_size)
                .join(marker_size, on='factor_name')
                .replace('MWTIX', 'TCW')
                )
    
    if drop_composites:
        mds_ts = mds_ts.query('composite == 0')

    if drop_trump:
        mds_ts = mds_ts.query('factor_name != "TRUMP"')
    
    if animate:
        fig = draw_mds_ts(mds_ts, tick_range=tick_range)
        fig.update_traces(textfont_color = 'lightgray')
    
    else:
        # print(dates)
        # print(dates[0])
        mds_latest = mds_ts[mds_ts['date'] == dates[0]].drop(columns='date')
        fig = draw_mds_ts(mds_latest, tick_range=tick_range)
        
        for i in range(len(dates) - 1):
            fig = add_whiskers(fig, mds_ts, dates[i], dates[i + 1])
        # fig = add_whiskers(fig, mds_ts, t0, t1)
        # fig = add_whiskers(fig, mds_ts, t1, t2)
        fig.update_layout(legend_title_text=None, title=f'{transformation_type[transformation]}')
        
                
        def get_trace_color(trace, legendgroup):
            return trace.marker.color if trace.legendgroup in legendgroup else 'lightgray'
        
        
        asset_class_list = ['Theme', 'Portfolio']
        fig.for_each_trace(lambda t: t.update(textfont_color = get_trace_color(t, asset_class_list)))
    
    # r = sqrt(2)/2
    # r = 0.8
    # fig.add_shape(
    #     type="circle",
    #     xref="x", yref="y",
    #     x0=-r, y0=-r, x1=r, y1=r,
    #     line_color='lightgray', line_width=.5,
    #     )
    
    return fig
