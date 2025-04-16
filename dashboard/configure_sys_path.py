import os
import sys


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

        # Optional: Keep debug prints to verify
        print("--- Debug Info ---")
        print(f"Current Working Directory: {os.getcwd()}")
        print("sys.path (after modification):")  # Added label
        for p in sys.path:
                print(f"  - {p}")
        print("--- End Debug Info ---")