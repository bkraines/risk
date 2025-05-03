import streamlit as st

from risk_lib.data import get_factor_data
from risk_lib.stats import get_dist_ma_set, get_days_ma_set
from risk_lib.chart import draw_volatility, draw_correlation, draw_cumulative_return, draw_volatility_ratio, draw_beta, draw_returns, draw_zscore, draw_distance_from_ma, draw_days_from_ma, draw_zscore_qq
from risk_lib.config import HALFLIFES
from dashboard.interface import select_date_range
from dashboard.interface import add_sidebar_defaults

def build_dashboard(factor_data):
    # TODO: Include time series of factor ratios
    
    factor_list = factor_data['factor_name'].values
    with st.sidebar:
        factor_1 = st.selectbox('Factor 1', options=factor_list, index=0)
        factor_2 = st.selectbox('Factor 2', options=factor_list, index=1)
        start_date, end_date = select_date_range(factor_data.indexes['date'], default_option='1y')
        vol_type = st.selectbox('Volatility Halflife for Z-Score', options=HALFLIFES, index=1)
        ma_type: int = st.number_input("Moving Average Window", value=200, min_value=1, step=1, format="%d")
    
    # Moving average must be computed before the date sliced
    # Can compute here or in `get_factor_data`
    factor_data['dist_ma'] = get_dist_ma_set(factor_data.cret, windows=[ma_type])
    factor_data['days_ma'] = get_days_ma_set(factor_data.cret, factor_data.vol, windows=[ma_type])
    # factor_data['zscore']  = get_zscore(factor_data.ret, factor_data.vol, shift=1)
    
    ds = factor_data.sel(date=slice(start_date, end_date))
    figs = {'cret':      draw_cumulative_return(ds.cret, factor_name=factor_1, factor_name_1=factor_2),
            'ret':       draw_returns(ds.ret, factor_name=factor_1, factor_name_1=factor_2),
            'zscore':    draw_zscore(ds.ret, ds.vol, factor_name=factor_1, factor_name_1=factor_2, vol_type=vol_type),
            'dist_ma':   draw_distance_from_ma(ds.dist_ma, factor_name=factor_1, factor_name_1=factor_2, window=ma_type),
            'days_ma':   draw_days_from_ma(ds.days_ma, factor_name=factor_1, factor_name_1=factor_2, window=ma_type, vol_type=vol_type),
            'corr':      draw_correlation(ds.corr, factor_name=factor_1, factor_name_1=factor_2, corr_type=HALFLIFES),
            'beta':      draw_beta(ds, factor_name=factor_1, factor_name_1=factor_2),
            'vol_1':     draw_volatility(ds.vol, factor_name=factor_1, vol_type=HALFLIFES),
            'vol_2':     draw_volatility(ds.vol, factor_name=factor_2, vol_type=HALFLIFES),
            'vol_ratio': draw_volatility_ratio(ds.vol, factor_name=factor_1, factor_name_1=factor_2, vol_type=HALFLIFES),
            'qqplot':    draw_zscore_qq(ds.ret, ds.vol, [(factor_1, vol_type), (factor_2, vol_type)]),
            }

    for fig in figs.values():
        st.plotly_chart(fig)

    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
