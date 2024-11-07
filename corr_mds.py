from typing import Union, Literal

from numpy import sqrt, cos, sin, arctan2, array
import xarray as xr
xr.set_options(keep_attrs=True,
               display_expand_data=False)
import pandas as pd
from sklearn.manifold import MDS

import plotly.express as px
import plotly.io as pio
# pio.renderers.default='png'
from plotly.graph_objs import Figure


def transform_coordinates(coordinates: pd.DataFrame, transformation_type=None, factor: str = 'SPY'):
    
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
    
    v = coordinates.loc[factor]
    x, y = v
        
    if transformation_type == 'rotate':
        theta = arctan2(y, x)
        return coordinates.pipe(rotate, -theta)
    
    if transformation_type == 'normalize':
        theta = arctan2(y, x)
        length = sqrt(sum(v**2))
        return coordinates.pipe(rotate, -theta).div(length)


def multidimensional_scaling(correlation_matrix: pd.DataFrame, init=None) -> pd.DataFrame:
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
    n_init = 4 if init is None else 1 
    embedding = MDS(dissimilarity='precomputed', random_state=42, n_init=n_init)
    coordinates = embedding.fit_transform(dissimilarity_matrix, init=init)
    
    return pd.DataFrame(coordinates, 
                        index=dissimilarity_matrix.index, 
                        columns=pd.Index(['dim1', 'dim2'], name='dimension'))


def mds_ts_df(corr: xr.DataArray, start_date = None, transformation=None, factor='SPY') -> pd.DataFrame:
    # TODO: Factor out corr_type
    dates = corr.sel(date=slice(start_date, None)).date.values
    coordinates = None
    
    mds_dict = {}
    for date in dates:
        df = corr.sel(date=date, corr_type=63).to_pandas()
        coordinates = multidimensional_scaling(df, init=coordinates)
        transformed = transform_coordinates(coordinates, transformation, factor=factor)
        mds_dict[date] = transformed
    return (pd.concat(mds_dict)
            .rename_axis(index=['date', df.index.name])
            )


def draw_mds_ts(df: pd.DataFrame, tick_range: Union[None, float, Literal['auto']] = 'auto') -> Figure:
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
    
    fig_animation = {'animation_frame': 'date', 'animation_group': 'asset'}  if 'date' in df.columns else {}
    fig_format    = {'template': 'plotly_white', 'height': 750, 'width': 750}
    
    
    fig = (px.scatter(df, 
                      x='dim1', y='dim2', text='asset', color='asset_class', 
                      **fig_animation,
                      **fig_format)
           .update_traces(textposition='middle right', textfont=dict(color='lightgray'))
           .update_xaxes(title=None)
           .update_yaxes(title=None)
           )
    
    if tick_range is not None:
        if tick_range == 'auto':
            tick_range = df[['dim1', 'dim2']].abs().max().max()
        print(tick_range)
        fig.update_xaxes(range=(-tick_range, tick_range)).update_yaxes(range=(-tick_range, tick_range))
        
    return fig
