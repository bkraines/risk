import streamlit as st
import pandas as pd

from data import get_factor_data
from chart import draw_volatility, draw_correlation
from util import check_memory_usage, summarize_memory_usage
from dashboard import build_streamlit_dashboard

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


halflifes = [21, 63, 126, 252, 512]
factor_data = get_factor_data(halflifes)

# tabs = st.tabs(["Correlation", "Feedback"])
# with tabs[0]:
#     build_dashboard_vol(factor_data)
# with tabs[1]:
#     build_streamlit_dashboard(factor_data)

build_dashboard_vol(factor_data)

del(factor_data)





# tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

# with tab1:
#     st.header("A cat")
#     st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
# with tab2:
#     st.header("A dog")
#     st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
# with tab3:
#     st.header("An owl")
#     st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

