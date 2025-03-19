from typing import Literal

IMAGE_DIR = 'images'
CACHE_DIR = 'cache'
CACHE_FILENAME = 'factor_data.zarr'
ARRAYLAKE_REPO = 'finance-demos/demo-icechunk'

CACHE_TARGET: Literal['disk', 'arraylake', 'streamlit'] = 'streamlit'
STREAMLIT_CACHE = False

# In case of `streamlit` caching, limit RAM usage by restricting `HALFLIFES`
# Unnecessary after `arraylake` caching is fully implemented
if STREAMLIT_CACHE or (CACHE_TARGET == 'streamlit'):
    HALFLIFES = [126]
else:
    HALFLIFES = [21, 63, 126, 252, 512]
