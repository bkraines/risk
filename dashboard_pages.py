import streamlit as st

from config import HALFLIFES

from data import build_factor_data
from feedback import build_streamlit_dashboard as market_feedback
from timeseries import build_dashboard_vol as time_series
from corr_mds_dashboard import build_streamlit_dashboard as corr_mds


factor_data = build_factor_data(HALFLIFES) #, streamlit=STREAMLIT_CACHE)

pg = st.navigation([st.Page(lambda: market_feedback(factor_data), 
                            title='Market Feedback',
                            url_path='feedback'), 
                    st.Page(lambda: time_series(factor_data), 
                            title='Time Series',
                            url_path='timeseries'),
                    st.Page(lambda: corr_mds(factor_data),
                            title='Correlation Projection',
                            url_path='correlation'),
                    ])

pg.run()

