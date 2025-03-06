import streamlit as st

pg = st.navigation([st.Page('feedback.py'), st.Page('timeseries.py')])
pg.run()