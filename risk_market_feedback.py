from itertools import product

from numpy import sqrt
import pandas as pd
import xarray as xr

from risk_data import get_factor_master
from risk_stats import total_return
from risk_chart import px_scatter, px_write
from risk_dates import format_date


# TODO: Add chart of showing correlation at t and t-n

def draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title=None, exclude_asset_classes=None):
    # TODO: Rename 'asset' to 'factor' in dataframe
    
    if exclude_asset_classes is None:
        exclude_asset_classes = []
    if return_title is None:
        return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'
    
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

    # factor_master = pd.DataFrame(factor_data['factor_name'].attrs).T
    factor_master = get_factor_master(factor_data)
    factor_master = factor_master[~factor_master['asset_class'].isin(exclude_asset_classes)]

    df = (pd.concat([corr, zscore, factor_master[['asset_class', 'hyper_factor', 'description']]], axis=1)
          # .replace('MWTIX', 'TCW')
          .rename_axis('asset').reset_index()
          # .replace('MWTIX', 'TCW')
        #   .assign(size = lambda df: df['hyper_factor'].apply(lambda x: 10 if x == 1 else 1).astype('float'))
          )
    
    # n = len(df.asset_class.unique())
    
    color_map_override = {'Portfolio': 'black', 
                          'Theme':     'red'}
    fig = (px_scatter(df, x='corr', y='zscore', text='asset', color='asset_class', #size='size',
                      color_map_override = color_map_override,
                      hover_data={'asset': True, 'description': True})
           .update_layout(yaxis_title=return_title,
                          xaxis_title=f'Correlation with {corr_asset}')
           .update_xaxes(showgrid=True, 
                         tick0=0,
                         dtick=0.25, 
                         zeroline=True,
                         zerolinecolor='lightgray')
           .update_yaxes(zeroline=True, 
                         zerolinecolor='lightgray')
           .update_layout(legend_orientation="h", 
                          legend_yanchor="top", 
                          legend_y=1.06, 
                        #   legend_entrywidthmode='fraction',
                        #   legend_entrywidth=1/(n+1),
                        #   legend_entrywidth=40,
                          ))
                        #   legend_xanchor="right", ) )
                        #   legend_x=1)
                    

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
        px_write(fig, f'feedback_{corr_asset}_{interval}.png')
        # px_write(fig, f'feedback_{corr_asset}_{interval}.html')
