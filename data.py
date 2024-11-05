from typing import List

import pandas as pd
import xarray as xr

import yfinance as yf

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
    #         .rename(index={'Ticker': 'asset'})
    #         .rename(columns=lambda col: col.lower(), level='ohlcv_type')
    #         .stack(['asset', 'ohlcv_type'], future_stack=True)
    #         .to_xarray())

    return ds


def get_factor_master(filename: str = 'factor_master.xlsx', sheet_name: str = 'read', index_col: str = 'factor_name') -> pd.DataFrame:
    df = pd.read_excel(filename, sheet_name=sheet_name, index_col=index_col)
    df.index = pd.CategoricalIndex(df.index, categories=df.index, ordered=True)
    return df