# app/import_setup.py
import sys
import os

# Add the parent directory of the project to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
