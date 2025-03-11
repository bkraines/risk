import streamlit as st
# from feedback import build_streamlit_dashboard as market_feedback
# from timeseries import build_dashboard_vol as time_series

pg = st.navigation([st.Page('feedback.py'), st.Page('timeseries.py')])
# pg = st.navigation([market_feedback, time_series])
pg.run()

