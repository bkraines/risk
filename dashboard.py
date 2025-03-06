import streamlit as st

pg = st.navigation([st.Page('dashboard.py'), st.Page('dashboard_vol.py')])
pg.run()