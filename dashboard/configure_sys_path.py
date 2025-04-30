import sys
import os
from pathlib import Path


def set_cwd_to_project_root(markers=["pyproject.toml", "requirements.txt", ".git", "PROJECT_ROOT"]):
    """
    Set the working directory to the project root using a list of possible markers.
    Tries each marker in order and stops when the first is found.

    Parameters
    ----------
    markers : list of str
        Files or directories that indicate the project root.
    """
    from pathlib import Path
    import os

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                os.chdir(parent)
                print(f"Working directory set to: {parent}")
                return
    raise FileNotFoundError(f"Project root not found (tried markers: {markers})")



def add_project_root_to_sys_path():
        """
        Sets up the project path by adding the project root directory to sys.path.
        """
        # Get the absolute path of the directory containing this script (dashboard)
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the absolute path of the project root directory (one level up)
        project_root = os.path.dirname(dashboard_dir)

        # Add the project root to the beginning of sys.path if it's not already there
        if project_root not in sys.path:
                sys.path.insert(0, project_root)

        # # Optional: Keep debug prints to verify
        # print("--- Debug Info ---")
        # print(f"Current Working Directory: {os.getcwd()}")
        # print("sys.path (after modification):")  # Added label
        # for p in sys.path:
        #         print(f"  - {p}")
        # print("--- End Debug Info ---")