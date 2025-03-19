IMAGE_DIR = 'images'
CACHE_DIR = 'cache'
ARRAYLAKE_REPO = 'finance-demos/demo-icechunk'

STREAMLIT_CACHE = False

if STREAMLIT_CACHE:
    HALFLIFES = [126] # [21, 63, 126, 252]
else:
    HALFLIFES = [21, 63, 126, 252, 512]

