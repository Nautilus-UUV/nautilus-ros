# UUV - ROS2 Stack

ROS2 navigation and control system for Nautilus's autonomous underwater vehicle, designed to run on RaspberryPi as part of the distributed software architecture.

## [Getting Started](docs/getting-started.md)

All about prerequisites, setup, development, package's architecture, testing and debugging.

## Packages

### `py_pkg` - Python ROS2 Package
Contains Python-based ROS2 nodes and auxiliary libraries:
- **[uuv_ros_core](src/py_pkg/py_pkg/uuv_ros_core/)** - Centralized topic definitions, QoS profiles, and node utilities
- Path planning, EKF filtering, PID control, MQTT communication

### `cpp_pkg` - C++ ROS2 Package
Prepared for C++ ROS2 nodes.


## Testing

Tests live in `src/py_pkg/test/`, organised in four tiers of increasing realism and cost:

| Tier | Location | What it exercises | Needs |
|---|---|---|---|
| 1 — unit | `test/unit/` | pure logic: PID, depth/ACU control system, physics, math | nothing |
| 2 — node | `test/node/` | in-process `rclpy` nodes with stubbed I/O via a `NodeHarness` | `rclpy` only |
| 3 — sim | `test/sim/` | end-to-end through the HAL into Gazebo, via `launch_testing`. Covers BCU bladder filling, ACU pitch/roll joints, the IMU → prefilter → EKF pose pipeline, the full closed-loop TRIM mission (in two variants — EKF in the loop and a sim-only ground-truth pose bridge swapped in for it), the full closed-loop SAWTOOTH cycle (single variant, EKF in the loop, asserting the bang-bang ACU pitch reaches both stroke endpoints in the right legs of one descend → ascend cycle to ~7.5 m), and the full closed-loop SURFACE mission (EKF in the loop, asserting the glider rises to gauge ~0 and `pathfinding_node` self-terminates back to IDLE). | full DAVE workspace built |
| 4 — HIL | `test/hil/` *(planned)* | same node code as Tier 2 against the real STM driver | bench hardware |

Tiers 3 and 4 are marker-gated (`@pytest.mark.sim`, `@pytest.mark.hil`) and excluded from default runs by `pytest.ini`.

Run from `src/py_pkg/`:

```bash
source /opt/ros/jazzy/setup.bash
pytest test/unit/ -v          # Tier 1
pytest test/node/ -v          # Tier 2
pytest test/                  # Tier 1 + 2 (sim/hil filtered out)
pytest -m sim test/sim/ -v    # Tier 3 — requires built DAVE workspace
```

### Digital Twin Integration Testing

To run integration testing of the whole system, follow the [nautilus-dave](https://github.com/Nautilus-UUV/nautilus-dave/tree/dev) repository Installation instructions, then drive the controller nodes (`bcu`, `acu`, `pid`) directly against the HAL bridge.


### Tier 3 sim tests — running with the GUI

See [running_sim.md](docs/running_sim.md) for detailed instructions on different dive profiles.

Tier 3 tests default to **headless** Gazebo (no window) so they run fast and don't need a display. To watch the world while a test runs, set the per-test env var and add `-s` (so pytest doesn't capture launch output):

```bash
source /opt/ros/jazzy/setup.bash
source /home/girji/dave_ws/install/setup.bash    # nautilus_hal + dave_demos

BCU_SIM_GUI=1         pytest -m sim test/sim/test_bcu_sim.py              -v -s
ACU_SIM_GUI=1         pytest -m sim test/sim/test_acu_pitch_sim.py        -v -s
ACU_SIM_GUI=1         pytest -m sim test/sim/test_acu_roll_sim.py         -v -s
EKF_SIM_GUI=1         pytest -m sim test/sim/test_ekf_pipeline_sim.py     -v -s
TRIM_SIM_GUI=1        pytest -m sim test/sim/test_trim_neutral_sim.py     -v -s
TRIM_GT_SIM_GUI=1     pytest -m sim test/sim/test_trim_neutral_sim_gt.py  -v -s
SAWTOOTH_SIM_GUI=1    pytest -m sim test/sim/test_sawtooth_sim.py         -v -s
SURFACE_SIM_GUI=1     pytest -m sim test/sim/test_surface_sim.py          -v -s
```

Most Tier 3 tests run in 30 s – 3 min; `test_sawtooth_sim` takes ~5 min because it exercises a full descend → ascend cycle to ~7.5 m and back. The ACU pitch axis is bang-bang on pressure error, so the SAWTOOTH test only asserts that each stroke endpoint shows up on `ACU_PITCH` during the matching leg — body-frame pitch attitude is not measured here, which keeps the test independent of the EKF orientation drift in `docs/ekf_node_issues.md`.

see [docs/running_sim.md](docs/running_sim.md).

`pytest --collect-only -m sim test/sim/` only enumerates the tests.


When adding a new node: write Tier 1 logic tests first, then a Tier 2 black-box node test using the `NodeHarness` pattern in `test/node/conftest.py`. Add a Tier 3 sim test only if the behavior depends on the full HAL → Gazebo path.

## [Contributing](docs/CONTRIBUTING.md)

Follow our contribution guidelines for development standards and workflows.

