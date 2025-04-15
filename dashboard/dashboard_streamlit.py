import streamlit as st

from risk_lib.data import get_factor_data
from dashboard.feedback import build_streamlit_dashboard as market_feedback
# from timeseries import build_dashboard_vol as time_series
# from corr_mds_dashboard import build_streamlit_dashboard as corr_mds
from dashboard.factor_master_dashboard import build_streamlit_dashboard as factor_master
# from risk_lib.util import add_sidebar_defaults

factor_data = get_factor_data()

pg = st.navigation([st.Page(lambda: market_feedback(factor_data), 
                            title='Market Feedback',
                            url_path='feedback'), 
                    
                    # st.Page(lambda: time_series(factor_data), 
                    #         title='Time Series',
                    #         url_path='timeseries'),
                    
                    # st.Page(lambda: corr_mds(factor_data),
                    #         title='Correlation Projection',
                    #         url_path='correlation'),
                    
                    st.Page(lambda: factor_master(),
                            title='Factor Master',
                            url_path='factor_master'),
                    ])

# add_sidebar_defaults()

pg.run()

