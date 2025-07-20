# from memory_profiler import profile
from typing import Iterable, Optional

from datetime import datetime
import pandas as pd
import xarray as xr

import yfinance as yf
import pandas_datareader.data as pdr

from risk_config import CACHE_TARGET, HALFLIFES, CACHE_FILENAME, FACTOR_FILENAME, FACTOR_DIR, FACTOR_SET
from risk_config_port import PORTFOLIOS
from risk_dates import business_days_ago #, latest_business_day
from risk_util import safe_reindex, cache, convert_df_to_json, trim_leading_zeros
from risk_stats import align_dates, calculate_returns_set, get_statistics_set, smart_dot
from risk_portfolios import build_all_portfolios, portfolio_weights_to_xarray


def get_yahoo_data(ticker, field_name, auto_adjust: bool = True) -> pd.Series:
    # TODO: Check cache first
    # cache.columns.get_level_values(1)
    return yf.download(ticker, auto_adjust=auto_adjust)[field_name].squeeze()


def get_yahoo_data_set(tickers: Iterable[str], 
                       field_name: str = 'Close', 
                       asset_names: Optional[Iterable[str]] = None, 
                       auto_adjust: bool = True, 
                       batch: bool = False) -> pd.DataFrame:
    # TODO: Consider renaming to get_yfinance_series_set
    # TODO: Troubleshoot any batching problems; 
    #       consider manually batching with parallelization 
    #       and `retry_with_backoff` decorator
    # TODO: Fix type error
    # TODO: If `batch=True` fails, revert to `batch=False`
    if asset_names is None:
        asset_names = tickers
    if batch:
        ticker_map = dict(zip(tickers, asset_names))
        df =  (yf.download(list(tickers), auto_adjust=auto_adjust, 
                           start='1900-01-01',
                           end=datetime.today().strftime('%Y-%m-%d'),)
               .xs(field_name, axis=1, level=0)
               .loc[:, tickers]
               .rename(columns=ticker_map)
               .rename_axis(index='date', columns='factor_name')
               )
    else:
        df = (pd.DataFrame({asset_name: get_yahoo_data(ticker, field_name) 
                            for asset_name, ticker in zip(asset_names, tickers)})
              .rename_axis(index='date', columns='factor_name'))
    return df



def get_fred_data_set(
    tickers: str | list[str],
    factor_names: Optional[str | list[str]] = None,
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None
) -> pd.DataFrame:
    """
    Downloads multiple FRED series as a DataFrame.

    Parameters:
        tickers (list[str]): list of FRED tickers (e.g., ["USEPUINDXD", "GDP"]).
        start (str | datetime | None): Start date (default: None = full history).
        end (str | datetime | None): End date (default: today).

    Returns:
        pd.DataFrame: DataFrame with datetime index and one column per ticker.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(factor_names, str):
        factor_names = [factor_names]
    if factor_names is None:
        factor_names = tickers
    return pd.concat({factor_name: pdr.DataReader(ticker, "fred", start=start, end=end).squeeze()
                      for factor_name, ticker in zip(factor_names, tickers)},
                     axis=1)


def read_factors_lmxl(file_name:  str = 'ThemeTracker_combined.xlsx', 
                      sheet_name: str = 'TimeSeries', 
                      index_col:  str = 'Date') -> pd.DataFrame:
    df = (pd.read_excel(file_name, sheet_name=sheet_name, index_col=index_col)
          .rename_axis(index='date', columns='factor_name'))
    df.columns = pd.CategoricalIndex(df.columns, ordered=True, categories=df.columns)
    return df


def get_lmxl_data_set(tickers: str | list[str],
                      factor_names: str | list[str] | None = None) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(factor_names, str):
        factor_names = [factor_names]
    if factor_names is None:
        factor_names = tickers
    ticker_to_factor_map = dict(zip(tickers, factor_names))

    df = read_factors_lmxl()
    return df[tickers].rename(columns=ticker_to_factor_map)


# # deprecated
# def get_yf_data(asset_list: list[str]) -> xr.DataArray:
#     # TODO: Name the returned datarray 'ohlcv'?
#     df = (yf.download(asset_list, auto_adjust=False)
#             .rename_axis(index='date', columns=['ohlcv_type', 'asset'])
#             .rename(index={'Ticker': 'asset'})
#             .rename(columns=lambda col: col.lower(), level='ohlcv_type')
#             # .stack(['asset', 'ohlcv_type'], future_stack=True) # Convert assetst to categorical first
#             .stack(['ohlcv_type'], future_stack=True)
#             .loc[:, asset_list]
#             )
#     df.columns = pd.CategoricalIndex(df.columns, categories=asset_list, ordered=True)
#     ds = df.stack().to_xarray()
#     assert isinstance(ds, xr.DataArray), f"Expected DataArray, got {type(ds)}"
#     return ds

# # deprecated
# def get_yf_returns(asset_list: list[str]) -> xr.Dataset:
#     # TODO: Combine with get_factor_data
#     ds = xr.Dataset()
#     ds['ohlcv'] = get_yf_data(asset_list)
#     ds['cret']  = ds['ohlcv'].sel(ohlcv_type='adj close')
#     ds['ret']   = ds['cret'].ffill(dim='date').pipe(xr_pct_change, 'date')
#     return ds


def get_portfolio_master(portfolios: dict):
    # TODO: Enable different parameters for each portfolio by looping through dictionary
    portfolio_master = pd.DataFrame(
        index=pd.CategoricalIndex(portfolios.keys(), name='factor_name'),
        data={
            'asset_class': 'Portfolio',
            'region': 'Portfolio', 
            'hyper_factor': 0,
            'composite': 0,
            # 'description': '',
            'source': 'portfolio',
            # 'ticker': '',
            'diffusion_type': 'normal',
            'multiplier': 0.0001
        }
    )
    return portfolio_master


def read_factor_master_xl(file_name: str = FACTOR_FILENAME, file_dir: str = FACTOR_DIR, sheet_name: str = FACTOR_SET, index_col: str = 'factor_name') -> pd.DataFrame:
    # FIXME: I moved to a flat directory structure
    # file_path = os.path.join(file_dir, file_name)
    # df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=index_col)
    df = pd.read_excel(file_name, sheet_name=sheet_name, index_col=index_col)
    df.index = pd.CategoricalIndex(df.index, categories=df.index, ordered=True)
    return df


def build_factor_master(portfolios: Optional[dict] = None, **kwargs) -> pd.DataFrame:
    factor_master = read_factor_master_xl(**kwargs)
    if portfolios is not None:
        portfolio_master = get_portfolio_master(portfolios)
        factor_master = (pd.concat([factor_master, portfolio_master])
                         .rename_axis(factor_master.index.name))
    return factor_master


def get_factor_master(factor_data: Optional[xr.Dataset | xr.DataArray] = None, portfolios: Optional[dict] = PORTFOLIOS, **kwargs) -> pd.DataFrame:
    """
    Retrieves factor master, either from `factor_data` if provided or directly from the Excel config file.

    Parameters
    ----------
    factor_data : Optional[xr.Dataset], default None
        If provided, the attributes of coordinate `factor_name` are extracted into a DataFrame.
    **kwargs : dict
        Additional keyword arguments optionally passed to `read_factor_master` to override defaults

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the factor master information, with 'factor_name' as a CategoricalIndex.
    """
    if factor_data is None:
        df = build_factor_master(portfolios=portfolios, **kwargs)
    else:
        # Read factor_master from data_set
        df = pd.DataFrame(factor_data.factor_name.attrs).T
        df.index = pd.CategoricalIndex(df.index, categories=df.index, ordered=True, name='factor_name')
    return df


def get_factor_composites(file_name: str = FACTOR_FILENAME, file_dir = FACTOR_DIR, sheet_name: str = 'read_composites', index_col: str = 'portfolio_name') -> pd.DataFrame:
    """Get composite factor definitions from Excel"""
    # FIXME: I moved to a flat directory structure
    # file_path = os.path.join(file_dir, file_name)
    # df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=index_col)
    df = pd.read_excel(file_name, sheet_name=sheet_name, index_col=index_col)
    df.index = pd.CategoricalIndex(df.index, categories=df.index.unique(), ordered=True)
    return (df.reset_index()
            .pivot(index='factor_name', 
                   columns='portfolio_name', 
                   values='weight')
            .fillna(0)
            )


def is_data_current(ds: xr.Dataset) -> bool:
    # TODO: If latest_date is a weekend, this will always return False
    date_latest = ds.indexes['date'].max().date()
    date_prior  = business_days_ago(1)
    return date_latest >= date_prior

    # # NEW ATTEMPT
    # latest_data_date = ds.indexes['date'].max().date()
    # latest_business_date = latest_business_day()
    # return latest_data_date >= latest_business_date


def get_levels(factor_master, source_function_map):
    levels_list = []
    for source, func in source_function_map.items():
        factor_list = factor_master.query(f"source=='{source}'").index
        if len(factor_list) == 0:
            continue
        print(f"Downloading {len(factor_list)} {source.upper()} factor{'s' if len(factor_list) != 1 else ''}")
        levels = func(tickers=factor_master.loc[factor_list, 'ticker'],
                      names=factor_list)
        levels_list.append(levels)
    return (pd.concat(levels_list, axis=1)
            .pipe(align_dates, ['SPY']))


@cache(CACHE_TARGET)
# @profile
def build_factor_data(halflifes: list[int], factor_set=FACTOR_SET, portfolios=PORTFOLIOS) -> xr.Dataset:
    # TODO: Consider renaming to `_get_factor_data`
    # TODO: Check vol units
    # TODO: Add timing to console logs
    # TODO: Enforce data variable and coordinate order in xarray Dataset
    # TODO: Replace 'composite==1' with 'source=="composite"' in factor_master
    #       (This is already done? Check that `factor_master['composite']` is not used elsewhere.)
    factor_master = get_factor_master(file_name='factor_master.xlsx', sheet_name=factor_set, portfolios=portfolios)
    diffusion_map = factor_master['diffusion_type']
    multiplier_map = factor_master['multiplier']

    # Get yfinance, FRED, and LMXL returns
    source_function_map = {
        'yfinance': lambda tickers, names: get_yahoo_data_set(asset_names=names.tolist(), tickers=tickers, batch=True),
        'fred':     lambda tickers, names: get_fred_data_set(factor_names=names, tickers=tickers),
        'lmxl':     lambda tickers, names: get_lmxl_data_set(factor_names=names, tickers=tickers)
    }
    levels = get_levels(factor_master, source_function_map)
    factor_returns = calculate_returns_set(levels,
                                           periods=1,
                                           diffusion_map=diffusion_map, 
                                           multiplier_map=multiplier_map)

    # Get composite returns 
    # `Composites` are those portfolios defined in factor_master.xlsx
    factor_list_composite = factor_master.query("source=='composite'").index
    print(f'Building {len(factor_list_composite)} composite factor{"s" if len(factor_list_composite) != 1 else ""}')
    
    # TODO: Multiply on full factor_returns, not just ret_yf
    factor_list_yf = factor_master.query("source=='yfinance'").index
    ret_yf = factor_returns[factor_list_yf]
    levels_yf = levels[factor_list_yf]
    
    if not factor_list_composite.empty:
        composite_weights = (get_factor_composites()
                            #   .loc[factor_list_composite]
                            .pipe(safe_reindex, factor_master)
                            .fillna(0)
                            .loc[factor_list_yf]
                            )

        composite_ret = smart_dot(ret_yf, composite_weights)
        factor_returns = pd.concat([factor_returns, composite_ret], axis=1)
    
    # Get portfolio returns:
    # `Portfolios` are those portfolios defined in risk_config_port.py
    factor_list_portfolios = factor_master.query("source=='portfolio'").index
    print(f'Building {len(factor_list_portfolios)} portfolio factor{"s" if len(factor_list_portfolios) != 1 else ""}')
    if not factor_list_portfolios.empty:
        rebalancing_dates = factor_returns.resample('M').last().index
        portfolio_returns, portfolio_weights_long = build_all_portfolios(portfolios, factor_returns, rebalancing_dates)
        factor_returns = pd.concat([factor_returns, 
                                    trim_leading_zeros(portfolio_returns[factor_list_portfolios])
                                    ], axis=1)

    print('Calculating factor statistics')
    levels_latest = levels_yf.iloc[-1]
    factor_data = get_statistics_set(factor_returns, factor_master, levels_latest, halflifes)
    
    if not factor_list_portfolios.empty:
        factor_data['portfolio_weights'] = portfolio_weights_to_xarray(portfolio_weights_long)
        # Zarr only accepts JSON-serializable attributes, but `portfolios` contains a pd.DataFrame:
        factor_data['portfolio_name'].attrs = convert_df_to_json(portfolios)
    print('Factor construction complete')
    return factor_data


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
