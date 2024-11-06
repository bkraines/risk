from typing import List

from numpy import sqrt
import pandas as pd
import xarray as xr

import yfinance as yf

from data import get_factor_master, get_yf_data
from util import xr_pct_change

def get_ewm_corr(ret: xr.DataArray, halflife: int) -> xr.DataArray:
    return (ret.to_pandas()
            .ewm(halflife=halflife)
            .corr()
            .rename_axis(columns=lambda name: f"{name}_1")
            .stack()
            .rename('corr')
            .to_xarray())


def get_volatility_set(ret: xr.DataArray, halflifes: List[int], dim: str='date') -> xr.DataArray:
    # TODO: Use daily 5-day returns
    return xr.concat([ret.rolling_exp({dim: h}, window_type='halflife').std() * sqrt(252) * 100
                      for h in halflifes],
                     dim=pd.Index(halflifes, name='vol_type'))


def get_correlation_set(ret: xr.DataArray, halflifes: List[int]) -> xr.DataArray:
    # TODO: Use daily 5-day returns
    return xr.concat([get_ewm_corr(ret, h) for h in halflifes],
                     dim=pd.Index(halflifes, name='corr_type'))


# def get_factor_data(asset_list: List[str], halflifes: List[int]) -> xr.Dataset:
def get_factor_data(halflifes: List[int] = None) -> xr.Dataset:
    if halflifes == None:
        halflifes = [21, 63, 126, 252, 512]
    
    factor_master = get_factor_master()
    asset_list = factor_master.index.to_list()
    
    ds = xr.Dataset()
    ds['ohlcv'] = get_yf_data(asset_list) #.to_dataset(name='ohlcv')
    ds['cret']  = ds['ohlcv'].sel(ohlcv_type='adj close')
    ds['ret']   = xr_pct_change(ds['cret'], 'date')
    ds['vol']   = get_volatility_set(ds['ret'], halflifes)
    ds['corr']  = get_correlation_set(ds['ret'], halflifes)
    
    ds['asset'].attrs = factor_master.T
    return ds