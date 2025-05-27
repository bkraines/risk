import streamlit as st

from risk_data import get_factor_data
from risk_stats import get_vix_regime, summarize_regime
from dashboard_interface import add_sidebar_defaults, select_date_window

def build_dashboard(factor_data):
    factor_list = factor_data['factor_name'].values
    
    with st.sidebar:
        factor_names = st.multiselect('Select Factors', options=factor_list, default=factor_list[:2])
        corr_factor_name = st.selectbox('Correlation Factor', options=factor_list, index=0)
        # factor_1 = st.selectbox('Factor 1', options=factor_list, index=0)
        # factor_2 = st.selectbox('Factor 2', options=factor_list, index=1)
        start_date, end_date = select_date_window(factor_data.indexes['date'], default_window_name='max')
        common_dates = st.toggle('Common Dates', value=True)

    ds = factor_data.sel(date=slice(start_date, end_date))
    vix_regime = get_vix_regime(ds.cret)
    ret = ds.ret.sel(factor_name=factor_names).to_pandas()
    if common_dates:
        start_date_common = ret.dropna().index.min()
        end_date_common = ret.dropna().index.max()
        ret = ret.loc[start_date_common:end_date_common]
    corr_factor_ret = ds.ret.sel(factor_name=corr_factor_name).to_pandas().rename(corr_factor_name)
    summary = summarize_regime(ret, groups=vix_regime, factor_corr=corr_factor_ret)
    st.write(f"VIX Regime Summary from {ret.index.min():%Y-%m-%d} to {ret.index.max():%Y-%m-%d}")
    st.dataframe(summary)

    add_sidebar_defaults()


if __name__ == "__main__":
    factor_data = get_factor_data()
    build_dashboard(factor_data)
    # add_sidebar_defaults()
    del(factor_data)
