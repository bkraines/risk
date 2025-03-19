from typing import Literal

IMAGE_DIR = 'images'
CACHE_DIR = 'cache'
ARRAYLAKE_REPO = 'finance-demos/demo-icechunk'

CACHE_TARGET: Literal['disk', 'arraylake', 'streamlit'] = 'streamlit'
STREAMLIT_CACHE = False

# Limit RAM usage until streamlit is hooked up to arraylake:
if STREAMLIT_CACHE or (CACHE_TARGET == 'streamlit'):
    HALFLIFES = [126]
else:
    HALFLIFES = [21, 63, 126, 252, 512]
