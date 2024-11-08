from typing import List

from numpy import sqrt
import pandas as pd
import xarray as xr

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


