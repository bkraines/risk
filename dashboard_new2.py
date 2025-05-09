from configure_sys_path import add_project_root_to_sys_path
add_project_root_to_sys_path()

import streamlit as st
from risk_data import get_factor_data

from dashboard_feedback import build_dashboard as feedback
from dashboard_timeseries import build_dashboard as timeseries
from dashboard_mds import build_dashboard as mds
from dashboard_master import build_dashboard as master
from dashboard_monitor import build_dashboard as monitor
from dashboard_volfitness import build_dashboard as volfitness

# from configure_sys_path import set_cwd_to_project_root
# set_cwd_to_project_root()

# from risk_util import add_sidebar_defaults

def create_pages(factor_data):
    """Returns a list of Streamlit pages configured with dashboards."""
    return [
        st.Page(lambda: feedback(factor_data), title='Market Feedback', url_path='feedback'),
        st.Page(lambda: timeseries(factor_data), title='Time Series', url_path='timeseries'),
        st.Page(lambda: mds(factor_data), title='Correlation Projection', url_path='correlation'),
        st.Page(lambda: master(factor_data), title='Factor Master', url_path='factor_master'),
        st.Page(lambda: monitor(factor_data), title='Performance', url_path='monitor'),
        st.Page(lambda: volfitness(factor_data), title='Vol Fitness', url_path='volfitness'),
        ]

def main():
    factor_data = get_factor_data()
    pages = create_pages(factor_data)
    pg = st.navigation(pages)
    # add_sidebar_defaults()
    pg.run()

if __name__ == "__main__":
    main()
