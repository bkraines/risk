from typing import List, Optional, TypeVar, Literal
from collections.abc import Mapping

from numpy import sqrt, nan
import pandas as pd
import xarray as xr

from risk_config import REGIME_DICT


def fill_returns(df):
    # Placeholder for more sophisticated logic
    # TODO: Don't fill sufficiently stale data
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

# ReturnCalculatorT = Callable[[PandasObjectT, int, float], PandasObjectT]
# (RETURN_CALCULATORS: Mapping[DiffusionType, ReturnCalculatorT] = 
#     {"lognormal": lambda cret, periods, multiplier: cret.pct_change(periods).div(multiplier),
#      "normal":    lambda cret, periods, multiplier: cret.diff(periods).div(multiplier),
#      })

PandasObjectT = TypeVar("PandasObjectT", pd.Series, pd.DataFrame)
DiffusionType = Literal["lognormal", "normal"]

def calculate_returns(cret: PandasObjectT, 
                      periods: int = 1, 
                      diffusion_type: Optional[DiffusionType] = None, 
                      multiplier: float = 1e-4,
                      ) -> PandasObjectT:
    match diffusion_type:
        case 'lognormal':
            return cret.pct_change(periods).div(multiplier)
        case 'normal':
            return cret.diff(periods).div(multiplier)
        # case 'normal10':
        #     return cret.diff().div(10)
        case _:
            raise ValueError(f'Unsupported diffusion_type of {diffusion_type} for {getattr(cret, "name", "unnamed factor")}')
        # case nan:
        #     raise ValueError(f'No diffusion_type provided for {cret.name}')


def calculate_returns_set(cret: pd.DataFrame, 
                          periods: int, 
                          diffusion_map: Mapping[str, DiffusionType], 
                          multiplier_map: Mapping[str, float],
                          ) -> pd.DataFrame:
    return (pd.DataFrame({factor: calculate_returns(cret[factor], 
                                                    periods, 
                                                    diffusion_map[factor], 
                                                    multiplier_map[factor]) 
                          for factor in cret.columns
                          })
            .rename_axis(index='date', columns='factor_name'))
    

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
    if multiplier_map is None:
        multiplier_map = {}
    return (pd.DataFrame({factor: accumulate_returns(ret = ret[factor], 
                                                     diffusion_type = diffusion_map[factor], 
                                                     level = level_map.get(factor, 100), 
                                                     multiplier = multiplier_map.get(factor, 1e-4)) 
                          for factor in ret.columns
                          })
            .rename_axis(index='date', columns='factor_name'))


def total_return(cret, date_start, date_end, 
                 diffusion_map: Optional[Mapping[str, DiffusionType]]=None, 
                 multiplier_map: Optional[Mapping[str, float]]=None):
    if diffusion_map is None and multiplier_map is None and cret.factor_name.attrs is not None:
        factor_master  = pd.DataFrame(cret.factor_name.attrs).T
        diffusion_map  = factor_master['diffusion_type']   # type: ignore[assignment]
        multiplier_map = factor_master['multiplier']       # type: ignore[assignment]
    # elif diffusion_map is None or multiplier_map is None:
    #     raise ValueError('diffusion_map and multiplier_map must be provided if factor_name.attrs is None')
    cret_interval = cret.sel(date=slice(date_start, date_end)).isel(date=[0, -1]).to_pandas()
    periods=1
    return calculate_returns_set(cret_interval, periods, diffusion_map, multiplier_map).iloc[-1]


# deprecate
def accumulate_returns_old(da_ret: xr.DataArray, dim: str = 'date') -> xr.DataArray:
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


def get_volatility_set_new(cret: xr.DataArray, halflifes: List[int], dim: str='date') -> xr.DataArray:
    # This is an attempt to build 5-day rolling volatilities
    periods = 1
    ret = calculate_returns(cret.to_pandas(), periods=periods).to_xarray()
    return xr.concat([(ret/100).rolling_exp({dim: h}, window_type='halflife').std() * sqrt(252 / periods)
                      for h in halflifes],
                     dim=pd.Index(halflifes, name='vol_type'))


def get_volatility_set(ret: xr.DataArray, halflifes: List[int], dim: str='date') -> xr.DataArray:
    # TODO: Use daily 5-day returns
    # FIXME: xr.rolling_exp() doesn't support min_periods
    return xr.concat([(ret/100).rolling_exp(window={dim: h}, window_type='halflife').std() * sqrt(252)
                      for h in halflifes],
                     dim=pd.Index(halflifes, name='vol_type'))


def get_ewm_corr(ret: xr.DataArray, halflife: int) -> xr.DataArray:
    return (ret.to_pandas()
            .ewm(halflife=halflife, min_periods=halflife*2)
            .corr() # TODO: Try min_periods = halflife * 2
            .rename_axis(columns=lambda name: f"{name}_1")
            .stack()
            .rename('corr')
            .to_xarray())


def get_correlation_set(ret: xr.DataArray, halflifes: List[int]) -> xr.DataArray:
    # TODO: Use daily 5-day returns
    return xr.concat([get_ewm_corr(ret, h) for h in halflifes],
                     dim=pd.Index(halflifes, name='corr_type'))


def get_beta_pair(vol: xr.DataArray, corr: xr.DataArray, factor_name: str, factor_name_1: str) -> xr.DataArray:
    # TODO: Currently assumes vol types and corr types match. Instead, pull cov_type dictionary
    # TODO: Just works on factor pair. Should work on full factor set
    vol_0 = vol.sel(factor_name=factor_name).rename({'vol_type': 'cov_type'})
    vol_1 = vol.sel(factor_name=factor_name_1).rename({'vol_type': 'cov_type'})
    corr  = corr.sel(factor_name=factor_name, factor_name_1=factor_name_1).rename({'corr_type': 'cov_type'})
    beta  = vol_1 / vol_0 * corr
    return beta.rename('beta')


def get_beta_set(vol: xr.DataArray, corr: xr.DataArray, cov_types: dict) -> None: # xr.DataArray:
    pass  # TODO: Implement this function


def get_zscore(ret: xr.DataArray, vol: xr.DataArray, shift=1) -> xr.DataArray:
    _ret = ret / 100
    _vol = vol / sqrt(252)
    zscore = _ret / _vol.shift({'date': shift})
    return zscore.rename('zscore')


def distance_from_moving_average(cret: xr.DataArray, window: int = 200, shift: int = 1) -> xr.DataArray:
    # TODO: Accomodate different diffusions with a `factor_return` function
    cret_ma = cret.rolling(date=window).mean()
    dist_ma = 100 * (cret / cret_ma.shift({'date': shift}) - 1)
    return dist_ma.rename('dist_ma')


def get_dist_ma_set(cret: xr.DataArray, windows: List[int]) -> xr.DataArray:
    return xr.concat([distance_from_moving_average(cret, w) 
                      for w in windows],
                     dim=pd.Index(windows, name='ma_type'))
    

def days_from_moving_average(cret: xr.DataArray, vol: xr.DataArray, window: int = 200, shift: int = 1) -> xr.DataArray:
    dist_ma = distance_from_moving_average(cret, window)
    daily_vol = vol / sqrt(252)
    days_ma = (dist_ma / daily_vol.shift({'date': shift}))
    return days_ma.rename('days_ma')


def get_days_ma_set(cret: xr.DataArray, vol: xr.DataArray, windows: List[int]) -> xr.DataArray:
    return xr.concat([days_from_moving_average(cret, vol, w) 
                      for w in windows],
                     dim=pd.Index(windows, name='ma_type'))


def get_vix_regime(cret: xr.DataArray, 
                   factor_name='^VIX3M', 
                   bins=REGIME_DICT['vix']['bins'], 
                   labels=REGIME_DICT['vix']['labels']
                   ) -> pd.Series:
    # TODO: Generalize to other regimes
    vix = cret.sel(factor_name=factor_name).to_series().dropna()
    vix_regime = pd.cut(vix,
                        bins=bins,
                        labels=labels,
                        right=False,  # left-closed: [a, b)
                        include_lowest=True  # include 0 in the first bin
                        )
    return vix_regime