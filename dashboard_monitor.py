import streamlit as st

from numpy import sqrt
import pandas as pd
import xarray as xr

from risk_data import get_factor_data, get_factor_master
from risk_dates import format_date
# from risk_market_feedback import draw_market_feedback_scatter
from risk_config import HALFLIFES
from dashboard_interface import add_sidebar_defaults, select_date_range

TABLE_CONFIG = (pd.DataFrame(columns =  ["variable",   "name",               "format"],
                             data    = [["vol",        "Volatility (% ann)", "%.1f"],
                                        ["zscore",     "Return (std)",       "%.1f"],
                                        ["ret",        "Return (%)",         "%.2f"],
                                        ["zscore_abs", "Return (|std|)",     "%.1f"],
                                        ]
                                ).set_index("variable"))


def build_monitor_table(factor_data: xr.Dataset, vol_type) -> pd.DataFrame:
    """Builds a monitor table for the given date."""
    
    # factor_data = factor_data.sel(date=slice(None, date))
    # vol_type = 63
    date_latest   = factor_data.date[-1].values
    date_incoming = factor_data.date[-2].values
    
    ret_latest = factor_data.ret.sel(date=date_latest).to_series().div(100).rename('ret')
    vol_incoming = factor_data.vol.sel(vol_type=vol_type, date=date_incoming).to_series().rename('vol')
    
    zscore = (ret_latest / (vol_incoming.mul(sqrt(1/252)))).rename('zscore')
    # zscore.to_frame().style

    # factor_master = factor_data.factor_name.attrs
    factor_master = pd.DataFrame(factor_data.factor_name.attrs).T
    factor_master.index = pd.CategoricalIndex(factor_master.index, categories=factor_master.index, ordered=True, name='factor_name')
    
    from numpy import abs
    df = (pd.concat([ret_latest, zscore, vol_incoming, abs(zscore).rename('zscore_abs')], axis=1)
        .join(factor_master[['asset_class', 'region']])
        .reset_index()
        .set_index(['asset_class', 'region', 'factor_name'])
        )

    latest_str = pd.Timestamp(date_latest).strftime('%Y-%m-%d')
    incoming_str = pd.Timestamp(date_incoming).strftime('%Y-%m-%d')
    df.name = f'Returns as of {latest_str}, vol as of {incoming_str}'
    
    return(df)
    

def build_dashboard(factor_data, table_config=None):
    # TODO: Use Ag-grid instead of streamlit for better customization
    # TODO: Incorporate column color into table_config. Generalize `style_zscore_abs`.

    if table_config is None:
        table_config = TABLE_CONFIG
    model_options = HALFLIFES
    model_default = model_options.index(63) if 63 in model_options else 0
    
    with st.sidebar:
        # corr_asset   = st.selectbox('Correlation Asset', options=factor_list, index=0)
        # return_start, return_end = select_date_range(factor_data.indexes['date'], default_option='MTD')
        vol_type     = st.selectbox('Volatility Halflife', options=model_options, index=model_default)
        # corr_type    = st.selectbox('Correlation Halflife', options=model_options, index=model_default)
    
    def style_zscore_abs(col):
        """Applies gray text color style to the zscore_abs column."""
        return ['color: lightgray'] * len(col) # You can also use 'lightgray' or a hex code like '#808080'
    
    df = build_monitor_table(factor_data, vol_type)
    st.write(df.name)
    styler = df.style.apply(style_zscore_abs, subset=['zscore_abs'])
    column_config = {column: st.column_config.NumberColumn(label=config["name"], 
                                                           format=config["format"])
                     for column, config in table_config.iterrows()}
    st.dataframe(styler, height=700, width=650, column_config=column_config)


    # # TODO: Add peak memory usage (before deleting factor_data)
    # # TODO: Aḍd initial memory usage (before loading factor_data)
    # factor_list = factor_data['factor_name'].values

    # model_options = HALFLIFES
    # model_default = model_options.index(126) if 126 in model_options else 0

    # with st.sidebar:
    #     corr_asset   = st.selectbox('Correlation Asset', options=factor_list, index=0)
    #     return_start, return_end = select_date_range(factor_data.indexes['date'], default_option='MTD')
    #     vol_type     = st.selectbox('Volatility Halflife', options=model_options, index=model_default)
    #     corr_type    = st.selectbox('Correlation Halflife', options=model_options, index=model_default)

    # return_title = f'Returns from {format_date(return_start)} to {format_date(return_end)} (std)'
    # fig = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, return_title)

    # st.write(fig)
    
    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
