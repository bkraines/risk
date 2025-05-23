from typing import Optional
import streamlit as st
import xarray as xr

from risk_data import get_factor_data
from risk_portfolios import build_all_portfolios
from risk_config_port import PORTFOLIOS
from risk_util import restore_df_from_json
from risk_chart_port import draw_portfolio_cret_df, draw_portfolio_weights, get_portfolio_summary, draw_portfolio_cret
from dashboard_interface import add_sidebar_defaults


def build_dashboard(factor_data: xr.Dataset):
    portfolio_names = factor_data['portfolio_name'].values
    summary = get_portfolio_summary(factor_data['ret']
                                    .sel(factor_name=portfolio_names)
                                    .to_pandas())
    fig1 = draw_portfolio_weights(factor_data.portfolio_weights)
    fig2 = draw_portfolio_cret(factor_data['cret'], portfolio_names)

    st.dataframe(summary)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    if factor_data['portfolio_name'].attrs is not None:
        st.write(restore_df_from_json(factor_data['portfolio_name'].attrs))
    add_sidebar_defaults()


def build_dashboard_old(factor_data: xr.Dataset):
    factor_returns = factor_data.ret.to_pandas()
    rebalancing_dates = factor_returns.resample('M').last().index
    portfolio_returns, portfolio_weights_long = build_all_portfolios(PORTFOLIOS, factor_returns, rebalancing_dates)

    summary = get_portfolio_summary(portfolio_returns)
    fig1 = draw_portfolio_cret_df(portfolio_returns)
    fig2 = draw_portfolio_weights(portfolio_weights_long)
    
    st.dataframe(summary)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    add_sidebar_defaults()


if __name__ == "__main__":
    # factor_data = get_factor_data()
    build_dashboard()
    # add_sidebar_defaults()
    # del(factor_data)
    
    