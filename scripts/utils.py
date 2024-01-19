import sys
import subprocess
from typing import List


def install_package(packages: List[str]):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])