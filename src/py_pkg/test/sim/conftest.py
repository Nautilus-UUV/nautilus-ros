"""Session-end gz reap for Tier 3 sim tests.

The per-test ``@post_shutdown_test`` reap only fires if launch_testing
finishes its shutdown phase — Ctrl-C or an early crash skips it and
leaves an orphaned gz-sim server behind. This fixture runs on pytest
teardown either way.
"""

import pytest

from ._sim_helpers import reap_lingering_gz


@pytest.fixture(scope="session", autouse=True)
def _reap_gz_at_session_end():
    yield
    reap_lingering_gz()
