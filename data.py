from typing import Optional, List

import pandas as pd
import xarray as xr

import yfinance as yf

# from data import get_factor_master, get_yf_data
from util import xr_pct_change, safe_reindex
from stats import align_dates, calculate_returns_set, accumulate_returns_set, get_volatility_set, get_correlation_set


def get_yahoo_data(ticker, field_name, cache=None):
    # TODO: Check cache first
    # cache.columns.get_level_values(1)
    return yf.download(ticker, auto_adjust=False)[field_name].squeeze()


def get_yahoo_data_set(tickers, field_name, asset_names=None):
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


def get_yf_returns(asset_list: List[str]) -> xr.Dataset:
    # TODO: Combine with get_factor_data
    ds = xr.Dataset()
    ds['ohlcv'] = get_yf_data(asset_list)
    ds['cret']  = ds['ohlcv'].sel(ohlcv_type='adj close')
    ds['ret']   = ds['cret'].ffill(dim='date').pipe(xr_pct_change, 'date')
    return ds


# deprecate
# def get_factor_data(asset_list: List[str], halflifes: List[int]) -> xr.Dataset:
def get_factor_data(halflifes: Optional[List[int]] = None) -> xr.Dataset:
    if halflifes == None:
        halflifes = [21, 63, 126, 252, 512]
    
    factor_master = get_factor_master(sheet_name='read_nocomposite')
    asset_list = factor_master.index.to_list()
    
    ds = xr.Dataset()
    ds['ohlcv'] = get_yf_data(asset_list) #.to_dataset(name='ohlcv')
    
    # ds['cret']  = ds['ohlcv'].sel(ohlcv_type='adj close')
    ds['ret']   = (ds['ohlcv']
                   .sel(ohlcv_type='adj close')
                   .ffill(dim='date')
                   .pipe(xr_pct_change, 'date'))
    ds['vol']   = get_volatility_set(ds['ret'], halflifes)
    ds['corr']  = get_correlation_set(ds['ret'], halflifes)
    
    ds['asset'].attrs = factor_master.T.to_dict()
    return ds

# deprecate
def build_dataset_with_composites(halflifes: List[int]) -> xr.Dataset:
    # Here `assets` refer to factors built from single security (basis vectors)
    # Here `portfolios` refer to factors built from `assets`
    # TODO: Rename to get_factor_data_with_composites
    
    factor_master = get_factor_master(sheet_name='read')
    asset_list = factor_master.loc[factor_master['composite'] == 0].index.to_list()
    asset_data = get_yf_returns(asset_list)
    asset_ret = asset_data['ret'].fillna(0).to_pandas()
    
    portfolios_weights = get_portfolios().pipe(safe_reindex, factor_master).fillna(0).loc[asset_list]  
    portfolios_ret = asset_ret @ portfolios_weights

    factor_data = xr.Dataset()
    factor_data['ret']   = pd.concat([asset_ret, portfolios_ret], axis=1).rename_axis(columns='asset')
    # factor_data['ret'] = (pd.concat([asset_ret['MWTIX'], portfolios_ret, asset_ret.drop(columns=['MWTIX'])], axis=1).rename_axis(columns='asset'))
    factor_data['vol']   = get_volatility_set(factor_data['ret'], halflifes)
    factor_data['corr']  = get_correlation_set(factor_data['ret'], halflifes)
    factor_data['cret']  = factor_data['ret'].cumsum(dim='date') # TODO: Drop zeros from the beginning

    factor_data['asset'].attrs = factor_master.T.to_dict()

    return factor_data


def build_factor_data2(halflifes: List[int], factor_set='read') -> xr.Dataset:
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



def align_indices(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align the indices of two DataFrames to ensure they can be joined without errors.
    
    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame.
    
    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        The two DataFrames with aligned indices.
    """
    common_index = df1.index.union(df2.index)
    return df1.reindex(common_index), df2.reindex(common_index)


# # Example usage:
# factor_data = get_factor_data()
# factor_ret = factor_data.ret.to_pandas().fillna(0)
# portfolios = get_portfolios()
# portfolio_ret = factor_ret @ portfolios

# # Align indices before joining
# factor_ret_aligned, portfolio_ret_aligned = align_indices(factor_ret, portfolio_ret)
# combined_ret = factor_ret_aligned.join(portfolio_ret_aligned, how='outer')