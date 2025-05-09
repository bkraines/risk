import streamlit as st

from risk_data import get_factor_master, get_factor_data

# factor_master = get_factor_master()
# st.dataframe(factor_master)

factor_data = get_factor_data(read_cache=False)
st.dataframe(factor_data.ret.to_pandas())