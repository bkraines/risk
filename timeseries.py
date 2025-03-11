import streamlit as st

from data import get_factor_data
from chart import draw_volatility, draw_correlation
from util import check_memory_usage, summarize_memory_usage
from config import STREAMLIT_CACHE

def build_dashboard_vol(factor_data):
    factor_list = factor_data['factor_name'].values

    with st.sidebar:
        factor_1 = st.selectbox('Factor 1', options=factor_list, index=1)
        factor_2 = st.selectbox('Factor 2', options=factor_list, index=2)

    figs = {'corr':  draw_correlation(factor_data.corr, factor_name=factor_1, factor_name_1=factor_2, corr_type=halflifes),
            'vol_1': draw_volatility(factor_data.vol, factor_name=factor_1, vol_type=halflifes),
            'vol_2': draw_volatility(factor_data.vol, factor_name=factor_2, vol_type=halflifes),
            }

    for fig in figs.values():
        st.write(fig)

    with st.sidebar:
        st.write(f'Memory usage: {check_memory_usage()} MB')
        st.table(summarize_memory_usage())


if STREAMLIT_CACHE:
    halflifes = [126] 
else:
    halflifes = [21, 63, 126, 252, 512]
factor_data = get_factor_data(halflifes, streamlit=STREAMLIT_CACHE)

build_dashboard_vol(factor_data)

del(factor_data)
