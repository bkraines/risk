import streamlit as st

from risk_data import get_factor_master

factor_master = get_factor_master()
st.dataframe(factor_master)