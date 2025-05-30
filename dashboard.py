# from configure_sys_path import add_project_root_to_sys_path #, set_cwd_to_project_root,
# add_project_root_to_sys_path()
# set_cwd_to_project_root()

import streamlit as st

from risk_data import get_factor_data
# from risk_util import add_sidebar_defaults

from dashboard_timeseries  import build_dashboard as time_series
from dashboard_correlation import build_dashboard as cross_section
from dashboard_feedback    import build_dashboard as feedback
from dashboard_mds         import build_dashboard as mds
from dashboard_monitor     import build_dashboard as monitor
from dashboard_volfitness  import build_dashboard as vol_fitness
from dashboard_portfolios  import build_dashboard as portfolios
from dashboard_master      import build_dashboard as master
from dashboard_regime      import build_dashboard as regime
from dashboard_event_study import build_dashboard as event_study

# TODO: Update streamlit `[theme]` section of `.streamlit/config.toml` file to match `plotly_white` colors

with st.spinner("Constructing factors..."):
    factor_data = get_factor_data()

pg = st.navigation([
                    st.Page(lambda: time_series(factor_data), 
                            title='Time Series',
                            url_path='time_series'),
                    
                    st.Page(lambda: cross_section(factor_data),
                            title='Cross Section',
                            url_path='cross_section'),

                    st.Page(lambda: feedback(factor_data), 
                            title='Market Feedback',
                            url_path='feedback'), 
                                        
                    st.Page(lambda: mds(factor_data),
                            title='Correlation Projection',
                            url_path='correlation'),

                    st.Page(lambda: event_study(factor_data),
                            title='Event Study',
                            url_path='event_study'),

                    st.Page(lambda: monitor(factor_data),
                            title='Performance',
                            url_path='monitor'),
                    
                    st.Page(lambda: vol_fitness(factor_data),
                            title='Vol Fitness',
                            url_path='vol_fitness'),

                    st.Page(lambda: regime(factor_data),
                            title='Regimes',
                            url_path='regimes'),
                    
                    st.Page(lambda: portfolios(factor_data),
                            title='Portfolios',
                            url_path='portfolios'),
                    
                    st.Page(lambda: master(factor_data),
                            title='Factor Master',
                            url_path='factor_master'),

                    ])

# add_sidebar_defaults()

pg.run()

