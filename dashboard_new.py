from configure_sys_path import add_project_root_to_sys_path
add_project_root_to_sys_path()

import streamlit as st
import pandas as pd

from risk_data import get_factor_data

from dashboard_feedback   import build_dashboard as feedback
from dashboard_timeseries import build_dashboard as timeseries
from dashboard_mds        import build_dashboard as mds
from dashboard_master     import build_dashboard as master
from dashboard_monitor    import build_dashboard as monitor
from dashboard_volfitness import build_dashboard as volfitness
# from risk_util import add_sidebar_defaults


# Dashboard configuration
dashboard_configs_df = pd.DataFrame(
    columns=['title',                  'url_path',      'fn'],
    data=  [['Market Feedback',        'feedback',      feedback],
            ['Time Series',            'timeseries',    timeseries],
            ['Correlation Projection', 'correlation',   mds],
            ['Factor Master',          'factor_master', master],
            ['Performance',            'monitor',       monitor],
            ['Vol Fitness',            'volfitness',    volfitness],
            ]
    )

def create_pages(factor_data):
    """Create Streamlit navigation pages based on config."""
    return [
        st.Page(lambda fn=row.fn: fn(factor_data), title=row.title, url_path=row.url_path)
        for _, row in dashboard_configs_df.iterrows()
    ]

def main():
    factor_data = get_factor_data()
    pages = create_pages(factor_data)
    # add_sidebar_defaults()
    st.navigation(pages).run()

if __name__ == "__main__":
    main()
