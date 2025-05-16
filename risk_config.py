from typing import Literal
from datetime import datetime

IMAGE_DIR = 'images'
CACHE_DIR = 'cache'
CACHE_FILENAME = 'factor_data.zarr'
ARRAYLAKE_REPO = 'finance-demos/demo-icechunk'
FACTOR_DIR = 'risk_lib'
FACTOR_FILENAME = 'factor_master.xlsx'
FACTOR_SET: Literal['read', 'read_short'] = 'read'    

CACHE_TARGET: Literal['disk', 'arraylake', 'streamlit'] = 'disk' #'streamlit'
STREAMLIT_CACHE = False

# In case of `streamlit` caching, limit RAM usage by restricting `HALFLIFES`
if STREAMLIT_CACHE or (CACHE_TARGET == 'streamlit'):
    HALFLIFES = [126]
else:
    HALFLIFES = [21, 63, 126, 252, 512]

COV_TYPES = {str(h): {'vol_type': h,
                      'corr_type': h}
             for h in HALFLIFES}

# Trailing windows
TRAILING_WINDOWS: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
    "3y": 252 * 3,
    "5y": 252 * 5,
    "10y": 252 * 10,
}

# Named historical event ranges
MARKET_EVENTS: dict[str, tuple[datetime, datetime]] = {
    "Trade War Selloff (2025)":  (datetime(2025,  3, 31), datetime(2025, 4, 21)),
    "Trade War Rally (2025)":    (datetime(2025,  4, 21), datetime(2025, 5, 16)),
    
    "GFC (2008)":           (datetime(2007, 10,  1), datetime(2009, 3,  9)),
    "Eurozone Crisis":      (datetime(2011,  7,  1), datetime(2012, 9,  1)),
    "Taper Tantrum (2013)": (datetime(2013,  5,  1), datetime(2013, 9,  1)),
    "China Deval (2015)":   (datetime(2015,  6,  1), datetime(2016, 2, 29)),
    "Trade Wars (2018)":    (datetime(2018,  1,  1), datetime(2019, 1,  1)),
    "COVID Crash":          (datetime(2020,  2, 15), datetime(2020, 3, 23)),
    "Inflation Shock":      (datetime(2022,  1,  1), datetime(2023, 6, 30)),
    "SVB Crisis":           (datetime(2023,  3,  1), datetime(2023, 4,  1)),

    "Brexit Referendum":       (datetime(2016, 6, 1), datetime(2016, 7, 15)),
    "Ukraine Invasion":        (datetime(2022, 2, 24), datetime(2022, 4, 30)),
    "Israel-Hamas War (2023)": (datetime(2023, 10, 7), datetime(2023, 12, 31)),

    "QE1":                          (datetime(2008, 11, 25), datetime(2010, 3, 31)),
    "QE2":                          (datetime(2010, 11, 3), datetime(2011, 6, 30)),
    "QE3":                          (datetime(2012, 9, 13), datetime(2014, 10, 29)),
    "Fed Hiking Cycle (2015–2018)": (datetime(2015, 12, 1), datetime(2018, 12, 31)),
    "Repo Liquidity (Sep 2019)":    (datetime(2019, 9, 16), datetime(2020, 1, 31)),
    "Aggressive Fed Hikes (2022)":  (datetime(2022, 3, 16), datetime(2023, 7, 31)),

    "BOJ Yield Curve Control Intro": (datetime(2016, 9, 21), datetime(2016, 12, 31)),
    "BOJ YCC Tweaks (2022)":         (datetime(2022, 12, 19), datetime(2023, 1, 31)),

    "ARK Mania":         (datetime(2020, 4, 1), datetime(2021, 2, 12)),
    "Tech Wreck (2022)": (datetime(2022, 1, 1), datetime(2022, 10, 15)),
} 
