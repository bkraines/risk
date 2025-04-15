from typing import Optional, List

import pandas as pd
import xarray as xr
xr.set_options(keep_attrs=True,
               display_expand_data=False)

import yfinance as yf

from risk_lib.util import xr_pct_change, safe_reindex
from risk_lib.data import get_factor_master, get_portfolios
from risk_lib.stats import get_volatility_set, get_correlation_set


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


def calculate_returns_old(cret, diffusion_type):
    match diffusion_type:
        case 'lognormal':
            return cret.pct_change().mul(10_000)
        case 'normal':
            return cret.diff().mul(100)
        # case 'normal10':
        #     return cret.diff().div(10)
        case _:
            raise ValueError(f'Unsupported diffusion_type of {diffusion_type} for {cret.name}')
        # case nan:
        #     raise ValueError(f'No diffusion_type provided for {cret.name}')


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


def accumulate_returns_old(ret, diffusion_type, level=None):
    # TODO: This drops the first observation
    if level is None:
        level = ret.iloc[-1]
    match diffusion_type:
        case 'lognormal':
            cret = ret.div(10_000).add(1).cumprod()
            cret = cret / cret.iloc[-1] * level
        case 'normal':
            cret = ret.div(100).cumsum()
            cret = cret - cret.iloc[-1] + level
        case _:
            raise ValueError(f'Unsupported diffusion_type of {diffusion_type} for {ret.name}')
    return cret


def accumulate_returns(ret, diffusion_type, level=None, multiplier=1e-4):
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
    return (pd.DataFrame({factor: accumulate_returns(ret = ret[factor], 
                                                     diffusion_type = diffusion_map[factor], 
                                                     level = level_map.get(factor, 100), 
                                                     multiplier = multiplier_map.get(factor, 1e-4)) 
                          for factor in ret.columns
                          })
            .rename_axis(index='date', columns='factor_name'))