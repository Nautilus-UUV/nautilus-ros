"""Skip-collection hook for Tier 3 (sim) and Tier 4 (hil) tests.

Stricter than the pytest.ini marker filter: pytest applies markers AFTER
importing modules, so without this hook a default ``pytest test/`` would
still pay the launch_testing import cost (and crash if it's missing).

Opt in with ``pytest -m sim`` or by targeting the dir (``pytest test/sim/``).
"""

import os
import sys


def _opted_in_for(marker: str, target_dir: str) -> bool:
    argv = " ".join(sys.argv)
    if "-m" in sys.argv:
        try:
            idx = sys.argv.index("-m")
            expr = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
            if marker in expr:
                return True
        except ValueError:
            pass
    if target_dir in argv:
        return True
    return False


_HERE = os.path.dirname(__file__)

collect_ignore = []
if not _opted_in_for("sim", "sim"):
    collect_ignore.append("sim")
if not _opted_in_for("hil", "hil"):
    collect_ignore.append("hil")
