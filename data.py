from typing import List

import pandas as pd
import xarray as xr

import yfinance as yf

from util import xr_pct_change, safe_reindex, business_days_ago, cache
from stats import align_dates, calculate_returns_set, accumulate_returns_set, get_volatility_set, get_correlation_set
from config import CACHE_TARGET, HALFLIFES, CACHE_FILENAME

def get_yahoo_data(ticker, field_name):
    # TODO: Check cache first
    # cache.columns.get_level_values(1)
    return yf.download(ticker, auto_adjust=False)[field_name].squeeze()


def get_yahoo_data_set(tickers, field_name, asset_names=None):
    # TODO: Possibly save time by sending yfinance full list of tickers instead of looping
    if asset_names is None:
        asset_names = tickers
    return (pd.DataFrame({asset_name: get_yahoo_data(ticker, field_name) 
                         for asset_name, ticker in zip(asset_names, tickers)})
            .rename_axis(index='date', columns='factor_name'))


# deprecated
def get_yf_data(asset_list: List[str]) -> xr.DataArray:
    # TODO: Name the returned datarray 'ohlcv'?
    df = (yf.download(asset_list, auto_adjust=False)
            .rename_axis(index='date', columns=['ohlcv_type', 'asset'])
            .rename(index={'Ticker': 'asset'})
            .rename(columns=lambda col: col.lower(), level='ohlcv_type')
            # .stack(['asset', 'ohlcv_type'], future_stack=True) # Convert assetst to categorical first
            .stack(['ohlcv_type'], future_stack=True)
            .loc[:, asset_list]
            )
    df.columns = pd.CategoricalIndex(df.columns, categories=asset_list, ordered=True)
    ds = df.stack().to_xarray()
    assert isinstance(ds, xr.DataArray), f"Expected DataArray, got {type(ds)}"
    return ds


def get_yf_returns(asset_list: List[str]) -> xr.Dataset:
    # TODO: Combine with get_factor_data
    ds = xr.Dataset()
    ds['ohlcv'] = get_yf_data(asset_list)
    ds['cret']  = ds['ohlcv'].sel(ohlcv_type='adj close')
    ds['ret']   = ds['cret'].ffill(dim='date').pipe(xr_pct_change, 'date')
    return ds


def get_factor_master(filename: str = 'factor_master.xlsx', sheet_name: str = 'read', index_col: str = 'factor_name') -> pd.DataFrame:
    df = pd.read_excel(filename, sheet_name=sheet_name, index_col=index_col)
    df.index = pd.CategoricalIndex(df.index, categories=df.index, ordered=True)
    return df


def get_portfolios(filename: str = 'factor_master.xlsx', sheet_name: str = 'read_composites', index_col: str = 'portfolio_name') -> pd.DataFrame:
    df = pd.read_excel(filename, sheet_name=sheet_name, index_col=index_col)
    df.index = pd.CategoricalIndex(df.index, categories=df.index.unique(), ordered=True)
    return (df.reset_index()
            .pivot(index='factor_name', 
                   columns='portfolio_name', 
                   values='weight')
            .fillna(0)
            )


def is_data_current(ds: xr.Dataset) -> bool:
    date_latest = ds.indexes['date'].max().date()
    date_prior  = business_days_ago(1)
    return date_latest >= date_prior


@cache(CACHE_TARGET)
def build_factor_data(halflifes: List[int], factor_set='read') -> xr.Dataset:
    # TODO: Consider renaming to `_get_factor_data`
    # TODO: Check vol units
    factor_master = get_factor_master('factor_master.xlsx', factor_set)
    factor_list = factor_master.index
    diffusion_map = factor_master['diffusion_type']
    multiplier_map = factor_master['multiplier']

    factor_list_yf = factor_master.query('source=="yfinance"').index
    levels_yf = (get_yahoo_data_set(asset_names = factor_list_yf.tolist(), 
                                     tickers = factor_master.loc[factor_list, 'ticker'],
                                     field_name = 'Adj Close')
                 .pipe(align_dates, ['SPY'])
                 )

    ret_yf = calculate_returns_set(levels_yf, 
                                   periods=1,
                                   diffusion_map=diffusion_map, 
                                   multiplier_map=multiplier_map)
    
    # factor_list_composite = factor_master.query('source==composite').index
    portfolios_weights = (get_portfolios()
                        #   .loc[factor_list_composite]
                          .pipe(safe_reindex, factor_master)
                          .fillna(0)
                          .loc[factor_list_yf]
                          )
    portfolios_ret = ret_yf @ portfolios_weights
    levels_latest = levels_yf.iloc[-1]

    factor_data = xr.Dataset()
    factor_data['ret']  = pd.concat([ret_yf, portfolios_ret], axis=1).rename_axis(columns='factor_name')
    factor_data['cret'] = accumulate_returns_set(factor_data['ret'].to_pandas(), diffusion_map, levels_latest, multiplier_map)
    factor_data['vol']  = get_volatility_set(factor_data['ret'], halflifes)
    factor_data['corr'] = get_correlation_set(factor_data['ret'], halflifes)
    factor_data['factor_name'].attrs = factor_master.T.to_dict()
    
    return factor_data #, diffusion_map, levels_latest


def get_factor_data(**kwargs) -> xr.Dataset:
    kwargs.setdefault("halflifes", HALFLIFES)
    match CACHE_TARGET:
        case 'disk':
            kwargs.setdefault("check", is_data_current)
            kwargs.setdefault("file_type", "zarr")
            kwargs.setdefault("cache_file", CACHE_FILENAME)
        case 'arraylake':
            kwargs.setdefault("check", is_data_current)
    return build_factor_data(**kwargs)


