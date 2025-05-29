import streamlit as st

from risk_config import HALFLIFES, VIX_COLORS
from risk_util import remove_items_from_list
from risk_stats import get_dist_ma_set, get_days_ma_set, get_vix_regime, summarize_regime
from risk_data import get_factor_data
from risk_chart import draw_volatility, draw_correlation, draw_cumulative_return, draw_volatility_ratio, draw_beta, draw_returns, draw_zscore, draw_distance_from_ma, draw_days_from_ma, draw_zscore_qq, add_regime_shading
from dashboard_interface import add_sidebar_defaults, select_date_window


def build_dashboard(factor_data):
    # TODO: Include time series of factor ratios
    
    factor_list = factor_data['factor_name'].values
    with st.sidebar:
        col1, col2 = st.columns([1, 1])
        factor_1 = col1.selectbox('Factor 1', options=factor_list, index=0)
        factor_2 = col2.selectbox('Factor 2', options=factor_list, index=1)
        
        start_date, end_date = select_date_window(factor_data.indexes['date'], default_window_name='1y')
        regime_shading = st.toggle('VIX Regime Shading', value=False)
        
        col3, col4 = st.columns([1, 1])
        vol_type = col3.selectbox('Volatility Halflife', options=HALFLIFES, index=0)
        ma_type: int = col4.number_input("Moving Average Window", value=200, min_value=1, step=20, format="%d")
    
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
    
    # regime_fig_list = ['cret', 'ret', 'zscore', 'dist_ma', 'days_ma']
    regime_fig_list = remove_items_from_list(figs.keys(), ['qqplot'])
    if regime_shading:
        vix_regime = get_vix_regime(ds.cret)
        for fig_name in regime_fig_list:
            figs[fig_name] = add_regime_shading(figs[fig_name], vix_regime, VIX_COLORS)

    # fig_name = {'cret':      'Cumulative Return',
    #             'ret':       'Daily Return',
    #             'zscore':    'Z-Score',
    #             'dist_ma':   'Distance from MA',
    #             'days_ma':   'Days from MA',
    #             'corr':      'Correlation',
    #             'beta':      'Beta',
    #             'vol_1':     f'Volatility {factor_1}',
    #             'vol_2':     f'Volatility {factor_2}',
    #             'vol_ratio': f'Volatility Ratio {factor_1} / {factor_2}',
    #             'qqplot':    f'QQ Plot {factor_1} / {factor_2}'}
  

    for fig in figs.values():
        st.plotly_chart(fig)

    if regime_shading:
        ret = factor_data.ret.sel(factor_name=[factor_1, factor_2]).to_pandas()
        summary = summarize_regime(ret, groups=vix_regime)
        st.dataframe(summary)

    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
