from typing import Optional, List

import pandas as pd
import xarray as xr

import yfinance as yf

# from data import get_factor_master, get_yf_data
from util import xr_pct_change
from stats import get_volatility_set, get_correlation_set


def get_yf_data(asset_list: List[str]) -> xr.DataArray:
    df = (yf.download(asset_list)
            .rename_axis(index='date', columns=['ohlcv_type', 'asset'])
            .rename(index={'Ticker': 'asset'})
            .rename(columns=lambda col: col.lower(), level='ohlcv_type')
            .stack(['ohlcv_type'], future_stack=True)
            .loc[:, asset_list]
            )
    df.columns = pd.CategoricalIndex(df.columns, categories=asset_list, ordered=True)
    ds = df.stack().to_xarray()

    # return (yf.download(asset_list)
    #         .rename_axis(index='date', columns=['ohlcv_type', 'asset'])
    #         .rename(index={'Ticker': 'asset'}))
    #         .rename(columns=lambda col: col.lower(), level='ohlcv_type')
    #         .stack(['asset', 'ohlcv_type'], future_stack=True)
    #         .to_xarray())

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


# def get_factor_data(asset_list: List[str], halflifes: List[int]) -> xr.Dataset:
def get_factor_data(halflifes: Optional[List[int]] = None) -> xr.Dataset:
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