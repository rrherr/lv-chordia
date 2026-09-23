import os
import sys
from .settings import *

# Use the package directory instead of current working directory
PACKAGE_PATH=os.path.dirname(os.path.abspath(__file__))
WORKING_PATH=os.path.dirname(os.path.dirname(PACKAGE_PATH))  # Go up two levels to package root

# The bundled checkpoints are package data (lv_chordia/config owns the location).
CACHE_DATA_PATH = os.path.join(os.path.dirname(PACKAGE_PATH), 'cache_data')

DEFAULT_DATA_STORAGE_PATH=DEFAULT_DATA_STORAGE_PATH.replace('$project_name$',os.path.basename(WORKING_PATH))
