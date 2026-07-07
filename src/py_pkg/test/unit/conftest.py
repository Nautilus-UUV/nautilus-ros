"""Shared Tier 1 fixtures: installed scenario-library lookups."""

import os

import pytest
from ament_index_python.packages import get_package_share_directory


@pytest.fixture
def library_scenario_path():
    """Resolve an installed py_pkg scenario-library YAML by filename."""

    def _path(name: str) -> str:
        return os.path.join(
            get_package_share_directory("py_pkg"),
            "scenarios",
            "library",
            name,
        )

    return _path


@pytest.fixture
def nominal_scenario_path(library_scenario_path) -> str:
    return library_scenario_path("nominal.yaml")
