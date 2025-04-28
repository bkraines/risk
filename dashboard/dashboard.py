from configure_sys_path import add_project_root_to_sys_path
add_project_root_to_sys_path()

import streamlit as st

from risk_lib.data import get_factor_data

from dashboard.dashboard_feedback   import build_dashboard as feedback
from dashboard.dashboard_timeseries import build_dashboard as timeseries
from dashboard.dashboard_mds        import build_dashboard as mds
from dashboard.dashboard_master     import build_dashboard as master
from dashboard.dashboard_monitor    import build_dashboard as monitor
# from risk_lib.util import add_sidebar_defaults

factor_data = get_factor_data()

pg = st.navigation([st.Page(lambda: feedback(factor_data), 
                            title='Market Feedback',
                            url_path='feedback'), 
                    
                    st.Page(lambda: timeseries(factor_data), 
                            title='Time Series',
                            url_path='timeseries'),
                    
                    st.Page(lambda: mds(factor_data),
                            title='Correlation Projection',
                            url_path='correlation'),
                    
                    st.Page(lambda: master(),
                            title='Factor Master',
                            url_path='factor_master'),
                    
                    st.Page(lambda: monitor(factor_data),
                            title='Performance',
                            url_path='monitor'),
                    ])

# add_sidebar_defaults()

pg.run()

