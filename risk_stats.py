# from memory_profiler import profile

from typing import Optional, TypeVar, Literal
from collections.abc import Mapping

from numpy import sqrt, nan, diag
import numpy as np
import pandas as pd
import xarray as xr

from risk_config import REGIME_DICT


def fill_returns(df):
    # Placeholder for more sophisticated logic
    # TODO: Don't fill sufficiently stale data
    #       Perhaps use ffill's limit parameter
    #
    return df.ffill()

# Unused
def pd_diag(ser: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(diag(ser), index=ser.index, columns=ser.index)

# Unused
def get_cov(vol: pd.Series, corr: pd.DataFrame) -> pd.DataFrame:
    """Construct covariance matrix from volatilities and correlations."""
    D = pd_diag(vol)
    return D @ corr @ D


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
            # FIXME: `pct_change` fails on empty pd.Series, like when yfinance donwload fails
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


def get_volatility_set_new(cret: xr.DataArray, halflifes: list[int], dim: str='date') -> xr.DataArray:
    # This is an attempt to build 5-day rolling volatilities
    periods = 1
    ret = calculate_returns(cret.to_pandas(), periods=periods).to_xarray()
    return xr.concat([(ret/100).rolling_exp({dim: h}, window_type='halflife').std() * sqrt(252 / periods)
                      for h in halflifes],
                     dim=pd.Index(halflifes, name='vol_type'))


def get_volatility_set(ret: xr.DataArray, halflifes: list[int], dim: str='date') -> xr.DataArray:
    # TODO: Use daily 5-day returns
    # FIXME: xr.rolling_exp() doesn't support min_periods
    return xr.concat([(ret/100).rolling_exp(window={dim: h}, window_type='halflife').std() * sqrt(252)
                      for h in halflifes],
                     dim=pd.Index(halflifes, name='vol_type'))


# import numpy as np
# import xarray as xr
# from xarray_einstats import stats
# from xarray_einstats.moving import ewm_meanvarcov

# def get_ewm_corr_new(ret: xr.DataArray, halflife: int) -> xr.DataArray:
#     """
#     Compute exponentially weighted correlation using xarray-einstats.

#     Parameters
#     ----------
#     ret : xr.DataArray
#         Return data with dims ('time', 'factor').
#     halflife : int
#         Halflife in days for exponential weighting.

#     Returns
#     -------
#     xr.DataArray
#         Correlation array with dims ('time', 'factor', 'factor_1').
#     """
#     alpha = 1 - np.exp(-np.log(2) / halflife)

#     # Compute EWM mean, var, cov
#     ewm = stats.moving.ewm_meanvarcov(ret, dim='time', alpha=alpha)
#     cov = ewm.covariance  # dims: ('time', 'factor', 'factor')

#     # Compute std dev for normalization
#     std = np.sqrt(xr.concat([ewm.variance.sel(factor=fac) for fac in ret.factor], dim='factor'))

#     # Compute correlation
#     denom = std.expand_dims(factor=ret.factor) * std.transpose('time', 'factor')
#     corr = xr.where(denom != 0, cov / denom, 0.0)

#     # Rename second factor dim for alignment with stacked output
#     return corr.rename({'factor': 'factor', 'factor_1': 'factor_1'})


# @profile
def get_ewm_corr(ret: xr.DataArray, halflife: int) -> xr.DataArray:
    return (ret.to_pandas()
            .astype(np.float32) # or normalize by stdev and go straight to 16-bit
            .ewm(halflife=halflife, min_periods=halflife*2)
            .corr()
            .astype(np.float16)
            .rename_axis(columns=lambda name: f"{name}_1")
            .stack()
            .rename('corr')
            .to_xarray())


# @profile
def get_correlation_set(ret: xr.DataArray, halflifes: list[int]) -> xr.DataArray:
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


def get_dist_ma_set(cret: xr.DataArray, windows: list[int]) -> xr.DataArray:
    return xr.concat([distance_from_moving_average(cret, w) 
                      for w in windows],
                     dim=pd.Index(windows, name='ma_type'))
    

def days_from_moving_average(cret: xr.DataArray, vol: xr.DataArray, window: int = 200, shift: int = 1) -> xr.DataArray:
    dist_ma = distance_from_moving_average(cret, window)
    daily_vol = vol / sqrt(252)
    days_ma = (dist_ma / daily_vol.shift({'date': shift}))
    return days_ma.rename('days_ma')


def get_days_ma_set(cret: xr.DataArray, vol: xr.DataArray, windows: list[int]) -> xr.DataArray:
    return xr.concat([days_from_moving_average(cret, vol, w) 
                      for w in windows],
                     dim=pd.Index(windows, name='ma_type'))


def get_vix_regime(cret: xr.DataArray, 
                   factor_name='^VIX', 
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


def get_statistics_set(factor_returns: pd.DataFrame, factor_master: pd.DataFrame, levels_latest: pd.DataFrame, halflifes: list[int]) -> xr.Dataset:
    # TODO: Accept list of statistics to calculate
    diffusion_map = factor_master['diffusion_type']
    multiplier_map = factor_master['multiplier']
    factor_data = xr.Dataset()
    print('Construct returns DataArray')
    factor_data['ret']  = factor_returns.rename_axis(columns='factor_name')
    print('Construct cumulative returns DataArray')
    factor_data['cret'] = accumulate_returns_set(factor_data['ret'].to_pandas(), diffusion_map, levels_latest, multiplier_map)
    print('Construct volatility DataArray')
    factor_data['vol']  = get_volatility_set(factor_data['ret'], halflifes)
    print('Construct correlation DataArray')
    factor_data['corr'] = get_correlation_set(factor_data['ret'], halflifes)
    print('Append metadata')
    factor_data['factor_name'].attrs = factor_master.T.to_dict()
    return factor_data


def summarize(df, 
              percentiles: Optional[list[float]]=[0.05, 0.5, 0.85], 
              factor_corr: Optional[pd.DataFrame]=None, 
              rfr=0.0, 
              freq=252):
    """Summarize a DataFrame of returns."""
    # TODO: Take DataArray of cret instead of DataFrame of ret
    # TODO: Calculate correlation on 5-day returns
    # TODO: Pass in list of outputs to include
    ann_mean = df.mean() * freq
    ann_std  = df.std() * sqrt(freq)
    
    # Compute correlation
    if factor_corr is None:
        corr = pd.DataFrame()
    else:
        # corr = df.corrwith(factor_corr)
        corr = df.corrwith(factor_corr).rename(f'corr {factor_corr.name}').to_frame().T
        
    # Compute quantiles
    if percentiles is None:
        quantiles = pd.DataFrame()
    else:
        quantiles = df.quantile(percentiles)
        quantiles.index = [f"p{int(p * 100):02d}" for p in percentiles]
    
    stats = {
        'sharpe':     (ann_mean - rfr) / ann_std,
        'mean':       df.mean(),
        'std':        df.std(),
        **corr.T.to_dict(),
        # 'skew':       df.skew(),
        # 'kurtosis':   df.kurtosis(),
        # 'min':        df.min(),
        **quantiles.T.to_dict(),
        # 'max':        df.max(),
        'count':      df.count(),
    }
    
    summary = pd.DataFrame(stats).T
    summary.index = pd.CategoricalIndex(
        summary.index, 
        categories=list(stats.keys()), 
        ordered=True,
        name='statistic'
    )
    return summary


def summarize_regime(df, groups=None, include_total=True, **kwargs):
    """Summarize returns by regime, optionally including total."""
    # TODO: Align regimes before summarizing
    # TODO: Loop through multiple regime groups

    summary = (df
               .groupby(groups, observed=True)
               .apply(summarize, **kwargs))

    if include_total:
        total_summary = summarize(df, **kwargs)
        # Align with the MultiIndex: (regime, statistic)
        total_summary.index = pd.MultiIndex.from_product(
            [['Total'], total_summary.index],
        )
        summary = pd.concat([summary, total_summary])

    # Inherit statistics index name from parent function
    statistics_name = summary.index.names[1]
    return (summary
            .rename_axis(['regime', statistics_name])
            .reorder_levels([statistics_name, 'regime'])
            .sort_index())
    
  
# import numpy as np
# def summarize_walrus(df):
#     # Keeping this just for the notable use of the `:=` walrus operator
#     return (
#         pd.concat(
#             stats := {'count': df.count(),
#                       'mean': df.mean() * 252,
#                       'std': df.std() * sqrt(252),
#                       'sharpe': pd.Series(
#                               np.where(df.std() != 0, (df.mean() / df.std()) * sqrt(252), np.nan),
#                           index=df.columns
#                       ),
#                       'min': df.min(),
#                       'max': df.max()
#                       }, axis=1)
#         .T
#         .set_index(pd.CategoricalIndex(stats.keys(), 
#                                        categories=stats.keys(), 
#                                        ordered=True)
#         )
#         .sort_index()
#     )


def smart_dot(returns: pd.DataFrame, composite_weights: pd.DataFrame) -> pd.DataFrame:
    """Matrix multiplication peformed for each composite factor avoid unnecessary NaNs."""
    # TODO: Generalize to arbitrary matrix multiplication
    dict_f = {}
    for factor, weights in composite_weights.items():
        weights_f = weights[weights!=0]
        returns_f = returns[weights_f.index].dropna()
        dict_f[factor] = returns_f @ weights_f
    return pd.concat(dict_f, axis=1)