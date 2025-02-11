from itertools import product

from numpy import sqrt
import pandas as pd
import xarray as xr

import plotly.express as px
import plotly.io as pio

from stats import total_return


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





def draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title):
    # TODO: Accomodate MTD, YTD, date interval, date range, and single day
    
    ndays = factor_data['cret'].sel(date=slice(return_start, return_end)).date.size - 1  # - 1
    corr  = (factor_data['corr']
             .shift(date=1)
             .sel(corr_type=corr_type, factor_name_1=corr_asset)
             .sel(date=return_start, method='ffill')
             .to_series()
             )
    vol   = (factor_data['vol']
             .shift(date=1)
             .sel(vol_type=vol_type)
             .sel(date=return_start, method='ffill')
             .to_series()
             .mul(sqrt(ndays / 252) * 100)
             )
    ret = total_return(factor_data['cret'], return_start, return_end)
    zscore = ret.div(vol).rename('zscore')

    factor_master = pd.DataFrame(factor_data['factor_name'].attrs).T

    df = (pd.concat([corr, zscore, factor_master['asset_class'], factor_master['hyper_factor']], axis=1)
          .replace('MWTIX', 'TCW')
          .rename_axis('asset').reset_index()
          .replace('MWTIX', 'TCW')
        #   .assign(size = lambda df: df['hyper_factor'].apply(lambda x: 10 if x == 1 else 1).astype('float'))
          )
    
    color_map_override = {'Portfolio': 'black', 
                          'Theme':     'red'}
    fig = (px_scatter(df, x='corr', y='zscore', text='asset', color='asset_class', #size='size',
                      color_map_override = color_map_override)
           .update_layout(yaxis_title=return_title,
                          xaxis_title=f'Correlation with {corr_asset}'))
    # from chart import px_format
    # fig = px_format(fig, 
    #                 x_title='Correlation with 10-year bond',
    #                 y_title='Return on Fed Announcement (std)')

    corr_min = df['corr'].min()
    fig.add_shape(
        type="line",
        x0=df[df['asset'] == corr_asset]['corr'].values[0] * corr_min,
        y0=df[df['asset'] == corr_asset]['zscore'].values[0] * corr_min,
        x1=df[df['asset'] == corr_asset]['corr'].values[0],
        y1=df[df['asset'] == corr_asset]['zscore'].values[0],
        line=dict(color='lightgray', width=2)
        )
    return fig


def get_intervals(ds: xr.Dataset) -> pd.DataFrame:
    # TODO: Accept list of intervals, say [5, 21, 63]
    dates = ds.indexes['date']
    date_latest = dates.max()
    year_start = pd.Timestamp(date_latest.year, 1, 1)
    month_start = pd.Timestamp(date_latest.year, date_latest.month, 1)
    ytd_start = dates[dates < year_start].max()  # prior_year_end
    mtd_start = dates[dates < month_start].max() # prior_month_end
    wow_start = dates[-5-1]                        # week_over_week
    
    return (pd.DataFrame(
                columns = ['name', 'start_date', 'end_date',   'chart_title'],
                data =   [['ytd',   ytd_start,    date_latest, 'Year-to-Date Return (std)'],
                          ['mtd',   mtd_start,    date_latest, 'Month-to-Date Return (std)'],
                          ['5d',    wow_start,    date_latest, '5-Day Return (std)'],
                          ])
            .set_index('name'))


def draw_market_feedback_scatter_set(factor_data, corr_asset_list, vol_type, corr_type):
    # TODO: Add output path
    # TODO: Add dates (and source) to chart
    intervals = get_intervals(factor_data)
    interval_list = intervals.index.to_list()

    for corr_asset, interval in product(corr_asset_list, interval_list):
        i = intervals.loc[interval]
        fig = draw_market_feedback_scatter(factor_data, i['start_date'], i['end_date'], vol_type, corr_type, corr_asset, i['chart_title'])
        fig.show(renderer='png')
        fig.write_image(f'feedback_{corr_asset}_{interval}.png')
        # fig.write_html(f'feedback_{corr_asset}_{interval}.html')
