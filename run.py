from __future__ import print_function

import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from s6e8.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
