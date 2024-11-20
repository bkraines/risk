from typing import Union, Literal

from numpy import sqrt, cos, sin, arctan2, array, identity
import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True,
               display_expand_data=False)
import pandas as pd
from sklearn.manifold import MDS

import plotly.express as px
import plotly.io as pio
# pio.renderers.default='png'
import plotly.graph_objects as go
# from plotly.graph_objs import Figure


def prepare_correlation(corr: xr.DataArray, transformation=None, start_date=None) -> pd.DataFrame:
    factor_master = pd.DataFrame(corr.asset.attrs).T
    return (mds_ts_df(corr, 
                      transformation=transformation, 
                      start_date=start_date)
            .reset_index()
            .join(factor_master, on='asset')
            .assign(date = lambda df: df['date'].astype(str)))


def transform_coordinates(coordinates: pd.DataFrame, transformation_type=None, factor: str = 'SPY', 
                          factor_list = None, coordinates_initial: pd.DataFrame= None) -> pd.DataFrame:
    
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
    
    if transformation_type == 'rotate':
        v = coordinates.loc[factor]
        x, y = v
        theta = arctan2(y, x)
        return coordinates.pipe(rotate, -theta)
    
    if transformation_type == 'rotate_initial':
            v = coordinates_initial.loc[factor]
            x, y = v
            theta = arctan2(y, x)
            return coordinates.pipe(rotate, -theta)
    
    if transformation_type == 'normalize':
        v = coordinates.loc[factor]
        x, y = v
        theta = arctan2(y, x)
        length = sqrt(sum(v**2))
        return coordinates.pipe(rotate, -theta).div(length)
    
    if transformation_type == 'rotate_list':
        thetas = {}
        if factor_list is None:
            factor_list = ['SPY', 'TLT']
        
        for factor in factor_list:
            v = coordinates.loc[factor]
            x, y = v
            thetas[factor] = arctan2(y, x)
        theta_avg = sum(thetas.values()) / len(thetas)
        return coordinates.pipe(rotate, -theta_avg)
        


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
    n_init = 100 # if init is None else 1 
    embedding = MDS(dissimilarity='precomputed', random_state=random_state, n_init=n_init)
    coordinates = embedding.fit_transform(dissimilarity_matrix, init=init)
    
    return pd.DataFrame(coordinates, 
                        index=dissimilarity_matrix.index, 
                        columns=pd.Index(['dim1', 'dim2'], name='dimension'))


def mds_ts_df(corr: xr.DataArray, start_date = None, transformation=None, factor='SPY', **kwargs) -> pd.DataFrame:
    # TODO: Factor out corr_type

    dates = corr.sel(date=slice(start_date, None)).date.values
    # coordinates = None
    transformed = None
    coordinates = None

    if transformation == 'rotate_initial':
        date = dates.max()
        df = corr.sel(date=date, corr_type=63).to_pandas() # * 0 + np.identity(len(corr.asset))
        coordinates = multidimensional_scaling(df, init=transformed) #init=coordinates) 
        transformed_initial = transform_coordinates(coordinates, 'rotate', factor='SPY')
        transformation = None
    else:
        transformed_initial = None
        
    # for date in dates:
    #     df = corr.sel(date=date, corr_type=63).to_pandas()
    #     coordinates = multidimensional_scaling(df, init=transformed) #init=coordinates) 
    #     transformed = transform_coordinates(coordinates, transformation, factor=factor, coordinates_initial=transformed_initial)
    #     mds_dict[date] = transformed
    #     return (pd.concat(mds_dict)
    #             .rename_axis(index=['date', df.index.name])
    #             )

  
    mds_dict = {}
    for date in dates:
        df = corr.sel(date=date, corr_type=63).to_pandas() # * 0 + np.identity(len(corr.asset))
        coordinates = multidimensional_scaling(df, init=transformed_initial) #init=transformed) 
        transformed = transform_coordinates(coordinates, transformation, factor=factor, factor_list=None, coordinates_initial=transformed_initial)
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
    
    args_animation = {'animation_frame': 'date', 'animation_group': 'asset'}  if 'date' in df.columns else {}
    args_format    = {'template': 'plotly_white', 'height': 750, 'width': 750}
    args_size      = {'size': 'size', 'size_max': 10} if 'size' in df.columns else {}
    args_textcolor = ['black' if condition else 'lightgray' for condition in (df['asset']=='SPY')]
    
    fig = (px.scatter(df, 
                      x='dim1', y='dim2', text='asset', color='asset_class', 
                      **args_size,
                      **args_animation,
                      **args_format)
           .update_traces(textposition='middle right', 
                          textfont_color='lightgray',
                        #   textfont_color=args_textcolor,
                          )
           .update_layout(xaxis_title=None,
                          yaxis_title=None,
                        #   xaxis_showticklabels=False,
                        #   yaxis_showticklabels=False,
                        #   xaxis_showgrid=False,
                        #   yaxis_showgrid=False,
                        #   xaxis_showline=True,
                        #   yaxis_showline=True,
                          xaxis_scaleanchor="y", 
                          xaxis_scaleratio=1,
                          yaxis_tickvals = [-0.5, 0, 0.5],
                          legend_title_text=None)
           )
    
    # fig.update_layout(yaxis_tickvals = fig)
    
    # Refactor this into separate function
    if tick_range is not None:
        if tick_range == 'auto':
            tick_range = df[['dim1', 'dim2']].abs().max().max()
        fig.update_xaxes(range=(-tick_range, tick_range)).update_yaxes(range=(-tick_range, tick_range))
    
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
    return fig