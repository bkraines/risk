import streamlit as st

from risk_lib.data import get_factor_data
from risk_lib.util import check_memory_usage, summarize_memory_usage
from risk_lib.config import CACHE_TARGET


def force_data_refresh():
    with st.sidebar:
        if st.button('Refresh Data'):
            get_factor_data(read_cache=False)
        st.write(f'Cache target: {CACHE_TARGET}')


def display_memory_usage():
    with st.sidebar:
        st.write(f'Memory usage: {check_memory_usage()} MB')
        st.table(summarize_memory_usage())


def add_sidebar_defaults():
    force_data_refresh()
    display_memory_usage()


if __name__ == "__main__":
    add_sidebar_defaults()