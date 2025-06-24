from risk_data import get_factor_data
import streamlit as st


def build_tradewar_report(factor_data):
    # from datetime import date
    from risk_market_feedback import draw_market_feedback_scatter
    from risk_dates import build_window_map
    from risk_config import ROLLING_WINDOWS, HISTORICAL_WINDOWS
    date_list = factor_data.indexes['date'].to_list()
    window_map = build_window_map(date_list, ROLLING_WINDOWS, HISTORICAL_WINDOWS)

    vol_type = corr_type = 63
    corr_asset_1, corr_asset_2 = 'SPY', 'DX-Y.NYB'
    exclude_asset_classes = ['Portfolio', 'Econ', 'Theme']
    title=None

    window_name = 'Trade War Selloff (2025)'
    return_start, return_end = window_map[window_name]
    corr_asset = corr_asset_1
    fig1 = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, title, exclude_asset_classes)
    st.write("In April's trade war selloff, nearly all factors performed in line with their S&P beta, effectively matching a one-factor CAPM model.")
    st.plotly_chart(fig1)
    corr_asset = corr_asset_2
    fig2 = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, title, exclude_asset_classes)
    st.write("Returns of the outlying factors, like Swiss franc, euro, yen, and gold, are well-explained by the dollar, as second driving factor")
    st.plotly_chart(fig2)

    window_name = 'Trade War Rally (2025)'
    return_start, return_end = window_map[window_name]
    corr_asset = corr_asset_1
    fig3 = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, title, exclude_asset_classes)
    corr_asset = corr_asset_2
    fig4 = draw_market_feedback_scatter(factor_data, return_start, return_end, vol_type, corr_type, corr_asset, title, exclude_asset_classes)
    st.write("The pattern held through May's trade reversal.")
    col1, col2 = st.columns([1, 1])
    col1.plotly_chart(fig3)
    col2.plotly_chart(fig4)
    
    
    from risk_chart import draw_cumulative_return
    from dashboard_interface import select_date_window
    
    start_date = '2024-04-01'
    end_date = '2025-06-18' # Change to today
    cret = factor_data.cret.sel(date=slice(start_date, end_date))
    fig = draw_cumulative_return(cret, factor_name='^TNX', factor_name_1='DX-Y.NYB')
    st.plotly_chart(fig)
    
    
    from risk_config import EVENT_STUDIES
    from risk_event_study import draw_event_study

    df = factor_data.ret.to_pandas() / 100
    event_list_1 = EVENT_STUDIES['Trade War — VIX selloff']
    event_list_2 = [('^VIX', t) for (_, t) in event_list_1]
    fig1 = draw_event_study(df, event_list=event_list_1)
    fig2 = draw_event_study(df, event_list=event_list_2, reverse_y_axis=True)

    st.write("To build a scenario, let's compare with the 2018 trade war.")
    st.plotly_chart(fig1)
    st.write("We picked the dates by aligning the initial VIX selloff.")
    st.plotly_chart(fig2)
    


def build_dashboard(factor_data=None):
    if factor_data is None:
        factor_data = get_factor_data()
    build_tradewar_report(factor_data)