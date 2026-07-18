"""Tier 3 parity: buoyancy.py constants == canonical model.sdf + world.

Sim-artifact test (house rule: SDF-reading tests are Tier 3 even
without Gazebo — marker-gated ``@pytest.mark.sim``). The correlated
buoyancy derivation in ``py_pkg.scenarios.buoyancy`` hardcodes the
glider's mass/volume budget with SDF provenance comments; this test
re-derives every number from the canonical ``model.sdf`` and
``dave_ocean_waves.world`` with ``xml.etree`` and asserts equality.
It is the tripwire against silent SDF/world edits skewing a sweep:
change a link mass, a collision box, the BuoyancyEngine clamps, or the
world density, and this fails before any scenario is sampled.

The dave sister repo may not be built (nautilus-ros-only CI): skip,
don't fail, when its packages are absent from the ament index. A
*resolvable* package with a missing file still fails loudly — a silent
skip there would disarm the tripwire.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from py_pkg.scenarios import buoyancy

pytestmark = pytest.mark.sim

# SDF spec: a <link> without <inertial><mass> defaults to 1.0 kg.
SDF_DEFAULT_LINK_MASS_KG = 1.0

# The links whose masses buoyancy.py treats as derivable trim, keyed to
# the module anchor each must match. Every other link sums into M_FIXED.
TRIM_LINKS = {
    "trimming_bow": buoyancy.NOMINAL_TRIM_BOW_KG,
    "trimming_stern": buoyancy.NOMINAL_TRIM_STERN_KG,
    "bladder_link": buoyancy.NOMINAL_TRIM_BLADDER_KG,
}


def _dave_share_path(package: str, *parts: str) -> Path:
    """Resolve a file inside a dave package's share dir (the same
    authority `nautilus_hal.render_sdf` spawns from; with the standard
    ``--symlink-install`` build these are the source files)."""
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )

    try:
        share = get_package_share_directory(package)
    except PackageNotFoundError:
        pytest.skip(f"dave sister repo not built ({package} not in ament index)")
    return Path(share).joinpath(*parts)


@pytest.fixture(scope="module")
def model() -> ET.Element:
    sdf = _dave_share_path(
        "dave_robot_models", "description", "glider_nautilus", "model.sdf"
    )
    return ET.parse(sdf).getroot().find("model")


@pytest.fixture(scope="module")
def links(model: ET.Element) -> dict[str, ET.Element]:
    return {link.get("name"): link for link in model.findall("link")}


def _link_mass(link: ET.Element) -> float:
    mass = link.find("inertial/mass")
    return SDF_DEFAULT_LINK_MASS_KG if mass is None else float(mass.text)


def _link_pose_x(link: ET.Element) -> float:
    pose = link.find("pose")
    return 0.0 if pose is None else float(pose.text.split()[0])


def _collision_box_volume(link: ET.Element) -> float:
    box = link.find("collision/geometry/box/size")
    assert box is not None, f"link {link.get('name')!r} has no collision box"
    x, y, z = (float(v) for v in box.text.split())
    return x * y * z


class TestFixedMassBudget:
    def test_fixed_link_mass_sum(self, links):
        # Document order matters for bitwise float-sum equality; the
        # module's literal sum is written in the same order.
        fixed_sum = 0.0
        for name, link in links.items():
            if name not in TRIM_LINKS:
                fixed_sum += _link_mass(link)
        assert fixed_sum == pytest.approx(buoyancy.M_FIXED_KG, abs=1e-12)

    def test_trim_link_masses_match_anchors(self, links):
        for name, anchor in TRIM_LINKS.items():
            assert _link_mass(links[name]) == anchor, name

    def test_fins_carry_implicit_default_inertials(self, links):
        # M_FIXED counts these at the SDF default; an explicit inertial
        # appearing on a fin means the budget must be re-derived.
        for name in ("LeftFin_link", "RightFin_link", "TopRudder_link"):
            assert links[name].find("inertial") is None, (
                f"{name} grew an explicit <inertial>; update "
                "buoyancy.M_FIXED_KG provenance"
            )


class TestStaticDisplacement:
    def test_exactly_three_collision_boxes(self, model):
        # The world graded-buoyancy plugin displaces every collision
        # geometry; V_STATIC_M3 assumes exactly these three boxes.
        collisions = model.findall("link/collision")
        assert len(collisions) == 3, [c.get("name") for c in collisions]

    def test_hull_box_volume(self, links):
        assert _collision_box_volume(links["base_link"]) == pytest.approx(
            buoyancy.V_HULL_BOX_M3, abs=1e-9
        )

    def test_bow_foam_volume(self, links):
        assert _collision_box_volume(links["trimming_bow"]) == pytest.approx(
            buoyancy.V_BOW_FOAM_M3, abs=1e-9
        )

    def test_stern_foam_volume(self, links):
        assert _collision_box_volume(links["trimming_stern"]) == pytest.approx(
            buoyancy.V_STERN_FOAM_M3, abs=1e-9
        )


class TestLeverArms:
    def test_hull_box_centred_at_origin(self, links):
        # buoyancy_centroid_x drops the hull term because the box sits
        # at x = 0 (link and collision pose both).
        assert _link_pose_x(links["base_link"]) == 0.0
        col_pose = links["base_link"].find("collision/pose")
        assert col_pose is None or float(col_pose.text.split()[0]) == 0.0

    def test_trim_link_x_poses(self, links):
        assert _link_pose_x(links["trimming_bow"]) == buoyancy.X_BOW_M
        assert _link_pose_x(links["trimming_stern"]) == buoyancy.X_STERN_M
        assert _link_pose_x(links["bladder_link"]) == buoyancy.X_BLADDER_M


class TestBuoyancyEngine:
    @pytest.fixture(scope="class")
    def engine(self, model) -> ET.Element:
        plugin = model.find("plugin[@name='gz::sim::systems::BuoyancyEngine']")
        assert plugin is not None, "BuoyancyEngine plugin missing from model.sdf"
        return plugin

    def test_max_volume_is_hard_clamp(self, engine):
        assert float(engine.find("max_volume").text) == (
            buoyancy.SDF_BLADDER_MAX_VOLUME_M3
        )

    def test_default_volume_is_nominal_spawn(self, engine):
        assert float(engine.find("default_volume").text) == (
            buoyancy.NOMINAL_SPAWN_VOLUME_M3
        )

    def test_neutral_volume_zero(self, engine):
        # The neutral condition assumes force ∝ rho * V with no offset.
        assert float(engine.find("neutral_volume").text) == 0.0

    def test_canonical_fluid_density(self, engine):
        # The canonical (untemplated) SDF prices the bladder at fresh
        # water — the density the nominal anchors were solved at.
        assert float(engine.find("fluid_density").text) == (
            buoyancy.WORLD_WATER_DENSITY_KGM3
        )


class TestWorldDensity:
    def test_default_density_matches_module(self):
        world = _dave_share_path("dave_worlds", "worlds", "dave_ocean_waves.world")
        root = ET.parse(world).getroot()
        density = root.find(".//graded_buoyancy/default_density")
        assert density is not None, "graded_buoyancy default_density missing"
        assert float(density.text) == buoyancy.WORLD_WATER_DENSITY_KGM3
