from typing import List, Optional

import pandas as pd
import xarray as xr

import yfinance as yf

from risk_config import CACHE_TARGET, HALFLIFES, CACHE_FILENAME, FACTOR_FILENAME, FACTOR_DIR, FACTOR_SET
from risk_dates import business_days_ago #, latest_business_day
from risk_util import xr_pct_change, safe_reindex, cache, convert_df_to_json, trim_leading_zeros
from risk_stats import align_dates, calculate_returns_set, accumulate_returns_set, get_volatility_set, get_correlation_set
from risk_config_port import PORTFOLIOS
from risk_portfolios import build_all_portfolios, portfolio_weights_to_xarray


def get_yahoo_data(ticker, field_name):
    # TODO: Check cache first
    # cache.columns.get_level_values(1)
    return yf.download(ticker, auto_adjust=False)[field_name].squeeze()


def get_yahoo_data_set(tickers, field_name, asset_names=None, batch=False):
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


@cache(CACHE_TARGET)
def build_factor_data(halflifes: List[int], factor_set=FACTOR_SET, portfolios=PORTFOLIOS) -> xr.Dataset:
    # TODO: Consider renaming to `_get_factor_data`
    # TODO: Check vol units
    # TODO: Refactor returns from yahoo, composites, and portfolios into separate functions
    # TODO: Add timing to console logs
    # TODO: Enforce data variable and coordinate order in xarray Dataset
    # TODO: Replace 'composite==1' with 'source=="composite"' in factor_master
    factor_master = get_factor_master(file_name='factor_master.xlsx', sheet_name=factor_set, portfolios=portfolios)
    factor_list = factor_master.index
    diffusion_map = factor_master['diffusion_type']
    multiplier_map = factor_master['multiplier']

    # Get yahoo returns:
    factor_list_yf = factor_master.query("source=='yfinance'").index
    print(f'Downloading {len(factor_list_yf)} yfinance factors')
    levels_yf = (get_yahoo_data_set(asset_names = factor_list_yf.tolist(), 
                                    tickers = factor_master.loc[factor_list_yf, 'ticker'],
                                    field_name = 'Adj Close')
                 .pipe(align_dates, ['SPY'])
                 )

    ret_yf = calculate_returns_set(levels_yf, 
                                   periods=1,
                                   diffusion_map=diffusion_map, 
                                   multiplier_map=multiplier_map)
    # ret_list = [ret_yf]
    factor_returns = ret_yf

    def smart_dot(returns: pd.DataFrame, composite_weights: pd.DataFrame) -> pd.DataFrame:
        """Matrix multiplication peformed for each composite factor avoid unnecessary NaNs."""
        # TODO: Generalize to arbitrary matrix multiplication
        dict_f = {}
        for factor, weights in composite_weights.items():
            weights_f = weights[weights!=0]
            returns_f = returns[weights_f.index].dropna()
            dict_f[factor] = returns_f @ weights_f
        return pd.concat(dict_f, axis=1)
    
    # Get composite returns 
    # `Composites` are those portfolios defined in factor_master.xlsx
    factor_list_composite = factor_master.query("source=='composite'").index
    print(f'Building {len(factor_list_composite)} composite factors')
    if not factor_list_composite.empty:
        composite_weights = (get_factor_composites()
                            #   .loc[factor_list_composite]
                            .pipe(safe_reindex, factor_master)
                            .fillna(0)
                            .loc[factor_list_yf]
                            )
        # composite_ret = ret_yf @ composite_weights
        composite_ret = smart_dot(ret_yf, composite_weights)
        # ret_list.append(portfolios_ret)
        factor_returns = pd.concat([factor_returns, composite_ret], axis=1)
    
    # Get portfolio returns:
    # `Portfolios` are those portfolios defined in risk_config_port.py
    factor_list_portfolios = factor_master.query("source=='portfolio'").index
    print(f'Building {len(factor_list_portfolios)} portfolio factors')
    if not factor_list_portfolios.empty:
        rebalancing_dates = factor_returns.resample('M').last().index
        portfolio_returns, portfolio_weights_long = build_all_portfolios(portfolios, factor_returns, rebalancing_dates)
        factor_returns = pd.concat([factor_returns, 
                                    trim_leading_zeros(portfolio_returns[factor_list_portfolios])
                                    ], axis=1)

    print('Calculating factor statistics')
    levels_latest = levels_yf.iloc[-1]
    factor_data = xr.Dataset()
    # factor_data['ret']  = pd.concat(ret_list, axis=1).rename_axis(columns='factor_name')
    factor_data['ret']  = factor_returns.rename_axis(columns='factor_name')
    factor_data['cret'] = accumulate_returns_set(factor_data['ret'].to_pandas(), diffusion_map, levels_latest, multiplier_map)
    factor_data['vol']  = get_volatility_set(factor_data['ret'], halflifes)
    factor_data['corr'] = get_correlation_set(factor_data['ret'], halflifes)
    factor_data['factor_name'].attrs = factor_master.T.to_dict()
    if not factor_list_portfolios.empty:
        factor_data['portfolio_weights'] = portfolio_weights_to_xarray(portfolio_weights_long)
        # Zarr only accepts JSON-serializable attributes, but `portfolios` contains a pd.DataFrame:
        factor_data['portfolio_name'].attrs = convert_df_to_json(portfolios)
    print('Factor construction complete')
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
