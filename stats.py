from typing import List

from numpy import sqrt, nan
import pandas as pd
import xarray as xr

from util import xr_pct_change


def fill_returns(df):
    return df.ffill()


def get_business_days(df, factor_names):
    return df[factor_names].dropna(how='any').index


def align_dates(df, business_day_factors):
    dates_raw = df.index
    dates_business = get_business_days(df, business_day_factors)
    dates_union = dates_raw.union(dates_business)
    return (df
            .reindex(dates_union)
            .pipe(fill_returns)
            .loc[dates_business])


def calculate_returns(cret, diffusion_type, multiplier=1e-4):
    match diffusion_type:
        case 'lognormal':
            return cret.pct_change().div(multiplier)
        case 'normal':
            return cret.diff().div(multiplier)
        # case 'normal10':
        #     return cret.diff().div(10)
        case _:
            raise ValueError(f'Unsupported diffusion_type of {diffusion_type} for {cret.name}')
        # case nan:
        #     raise ValueError(f'No diffusion_type provided for {cret.name}')


def calculate_returns_set(df, diffusion_map, multiplier_map):
    return (pd.DataFrame({factor: calculate_returns(df[factor], diffusion_map[factor], multiplier_map[factor]) 
                          for factor in df.columns
                          })
            .rename_axis(index='date', columns='factor_name'))
    

def accumulate_returns_new(ret, diffusion_type, level=None, multiplier=1e-4):
    # TODO: This drops the first observation
    if level is None:
        level = ret.iloc[-1]
    match diffusion_type:
        case 'lognormal':
            cret = ret.mul(multiplier).add(1).cumprod()
            cret = cret / cret.iloc[-1] * level
        case 'normal':
            cret = ret.mul(multiplier).cumsum()
            cret = cret - cret.iloc[-1] + level
        case _:
            raise ValueError(f'Unsupported diffusion_type of {diffusion_type} for {ret.name}')
    return cret


def accumulate_returns_set(ret, diffusion_map, level_map=None, multiplier_map=None):
    if level_map is None:
        level_map = {factor: None for factor in ret.columns}  
    return (pd.DataFrame({factor: accumulate_returns_new(ret = ret[factor], 
                                                     diffusion_type = diffusion_map[factor], 
                                                     level = level_map.get(factor, 100), 
                                                     multiplier = multiplier_map.get(factor, 1e-4)) 
                          for factor in ret.columns
                          })
            .rename_axis(index='date', columns='factor_name'))



# deprecate
def accumulate_returns(da_ret: xr.DataArray, dim: str = 'date') -> xr.DataArray:
    """ 
    Cumulate returns along a specified dimension, handling NaNs appropriately.

    Parameters
    ----------
    da_ret : xarray.DataArray
        The input data array containing returns to be cumulated.
    dim : str, optional
        The dimension along which to cumulate the returns. Default is 'date'.

    Returns
    -------
    xarray.DataArray
        A data array with cumulated returns along the specified dimension.
        NaNs are preserved at the top of the array but forward-filled after.

    """
    # TODO: Specify which assets should act as calendar,
    #       e.g. drop days if SPY or IEF is missing
    return (da_ret.cumsum(dim=dim)
            .where(~da_ret.isnull(), nan)
            .ffill(dim=dim)
    )


def get_volatility_set(ret: xr.DataArray, halflifes: List[int], dim: str='date') -> xr.DataArray:
    # TODO: Use daily 5-day returns
    return xr.concat([(ret/100).rolling_exp({dim: h}, window_type='halflife').std() * sqrt(252)
                      for h in halflifes],
                     dim=pd.Index(halflifes, name='vol_type'))


def get_ewm_corr(ret: xr.DataArray, halflife: int) -> xr.DataArray:
    return (ret.to_pandas()
            .ewm(halflife=halflife)
            .corr() # TODO: Try min_periods = halflife * 2
            .rename_axis(columns=lambda name: f"{name}_1")
            .stack()
            .rename('corr')
            .to_xarray())


def get_correlation_set(ret: xr.DataArray, halflifes: List[int]) -> xr.DataArray:
    # TODO: Use daily 5-day returns
    return xr.concat([get_ewm_corr(ret, h) for h in halflifes],
                     dim=pd.Index(halflifes, name='corr_type'))


