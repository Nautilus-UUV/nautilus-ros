"""Tier 3: persistent BCU faults end-to-end through the in-process bridge.

In-process bridge tests on the shared ``_bcu_bridge_harness`` rig (no
Gazebo boot — seconds, not minutes), one rig per fault class:

- pump fault: ``fault_effectiveness:=0.5`` scales the commanded RPM
  through the real PumpDynamics transient — the feedback echo must
  plateau at exactly half the command, never above it, and the
  ``/bcu/rpm/fault`` provenance stream must read the constant 0.5;
- tank sensor fault: ``tank_fault_kind:=stuck`` freezes the *reported*
  tank pressure on its quantization comb while the underlying bladder
  volume sweeps the full span;
- comms fault: ``comms_drop_prob:=1.0`` silences every bridged
  telemetry stream while the Gazebo-facing plant path (BUOYANCY_COMMAND)
  and the ungated fault-provenance stream keep flowing — proving the
  gate wraps exactly the link, not the plant.

Don't run alongside any other sim/rclpy process on the host — the
production topic names overlap.
"""

import pytest
from std_msgs.msg import Float32, Float64, Int16, Int32, UInt8

from ._sim_helpers import spin_for, spin_until

pytestmark = pytest.mark.sim

# The dave sister repo may not be built (nautilus-ros-only CI): skip,
# don't fail, exactly like test_buoyancy_budget_parity.
pytest.importorskip("nautilus_hal")

from nautilus_hal.constants import SimDebugTopics, SimTopics  # noqa: E402
from py_pkg.robot_specs import BCU_MOTOR_VALVE_MASK  # noqa: E402
from py_pkg.uuv_ros_core import (  # noqa: E402
    UUVTopics,
    create_publisher_for_topic,
    create_subscription_for_topic,
)

from ._bcu_bridge_harness import (  # noqa: E402
    HELD_RPM,
    PLANT,
    SEED_VOLUME_M3,
    SIM,
    BridgeProbe,
    make_rig,
    wait_wired,
)

EFFECTIVENESS = 0.5
# Half command after the fault scaling; PumpDynamics' slew clamps onto
# the target exactly, so the echo plateau is exact.
EXPECTED_PLATEAU = int(HELD_RPM * EFFECTIVENESS)
# Dead time + slew-up to the (faulted) target + generous slack.
PLATEAU_BUDGET_S = (
    PLANT.pump_response_delay_s + EXPECTED_PLATEAU / PLANT.pump_slew_rpm_per_s + 6.0
)


class _FaultProbe(BridgeProbe):
    """Adds tank-pressure, fault-provenance, and valve-command taps."""

    def __init__(self):
        super().__init__("bcu_fault_probe")
        self.pressure_samples: list[int] = []
        self.fault_samples: list[float] = []
        self.valves_pub = create_publisher_for_topic(self, UUVTopics.BCU_VALVES)
        create_subscription_for_topic(self, UUVTopics.BCU_PRESSURE, self._on_pressure)
        self.create_subscription(
            Float32, SimDebugTopics.BCU_PUMP_FAULT, self._on_fault, 10
        )

    def _on_pressure(self, msg: Int32) -> None:
        self.pressure_samples.append(int(msg.data))

    def _on_fault(self, msg: Float32) -> None:
        self.fault_samples.append(float(msg.data))


class _CommsProbe(_FaultProbe):
    """Adds every remaining bridged stream plus the Gazebo-facing tap."""

    def __init__(self):
        super().__init__()
        self.volume_samples: list[int] = []
        self.flow_samples: list[float] = []
        self.valve_fb_samples: list[int] = []
        self.buoyancy_cmds: list[float] = []
        create_subscription_for_topic(
            self, UUVTopics.BCU_VOLUME, lambda m: self.volume_samples.append(m.data)
        )
        create_subscription_for_topic(
            self, UUVTopics.BCU_FLOW_RATE, lambda m: self.flow_samples.append(m.data)
        )
        create_subscription_for_topic(
            self,
            UUVTopics.BCU_FEEDBACK_VALVES,
            lambda m: self.valve_fb_samples.append(m.data),
        )
        self.create_subscription(
            Float64,
            SimTopics.BUOYANCY_COMMAND.format(model_name=SIM.model_name),
            lambda m: self.buoyancy_cmds.append(float(m.data)),
            10,
        )


rig_effectiveness = make_rig(
    _FaultProbe,
    init_args=["--ros-args", "-p", f"fault_effectiveness:={EFFECTIVENESS}"],
    name="rig_effectiveness",
)
rig_stuck = make_rig(
    _FaultProbe,
    init_args=["--ros-args", "-p", "tank_fault_kind:=stuck"],
    name="rig_stuck",
)
rig_comms = make_rig(
    _CommsProbe,
    init_args=["--ros-args", "-p", "comms_drop_prob:=1.0", "-p", "comms_seed:=7"],
    name="rig_comms",
)


def test_pump_effectiveness_scales_feedback(rig_effectiveness):
    probe, executor = rig_effectiveness
    wait_wired(probe, executor)

    # Held command (STM semantics: it stands until superseded).
    probe.valves_pub.publish(UInt8(data=BCU_MOTOR_VALVE_MASK))
    probe.rpm_pub.publish(Int16(data=HELD_RPM))

    def _at_plateau() -> bool:
        recent = [v for _, v in probe.fb_samples[-5:]]
        return len(recent) == 5 and all(v == EXPECTED_PLATEAU for v in recent)

    assert spin_until(executor, _at_plateau, timeout_s=PLATEAU_BUDGET_S), (
        f"echo never plateaued at {EXPECTED_PLATEAU};"
        f" trailing: {[v for _, v in probe.fb_samples][-10:]}"
    )
    # The faulted plant can never exceed effectiveness * command.
    peak = max(v for _, v in probe.fb_samples)
    assert peak <= EXPECTED_PLATEAU, f"echo overshot the faulted target: {peak}"
    # Provenance stream carries the constant effectiveness.
    assert probe.fault_samples, "no /bcu/rpm/fault samples"
    assert all(v == pytest.approx(EFFECTIVENESS) for v in probe.fault_samples)


def test_tank_stuck_freezes_pressure_while_input_sweeps(rig_stuck):
    probe, executor = rig_stuck
    wait_wired(probe, executor)

    # Sweep the bladder-volume stand-in across the full operating span —
    # the un-faulted tank map would traverse ~92 kPa, so a constant
    # output stream cannot be vacuous.
    steps = 20
    for i in range(steps):
        frac = i / (steps - 1)
        volume = PLANT.bladder_min_m3 + frac * (
            PLANT.bladder_max_m3 - PLANT.bladder_min_m3
        )
        probe.volume_state_pub.publish(Float64(data=volume))
        spin_for(executor, 0.15)

    assert len(probe.pressure_samples) >= 10, "too few pressure samples"
    assert len(set(probe.pressure_samples)) == 1, (
        "stuck tank pressure did not freeze:"
        f" {sorted(set(probe.pressure_samples))[:5]}..."
    )
    # The latched value is the first *reported* sample: post-noise, on
    # the 600 Pa quantization comb.
    assert probe.pressure_samples[0] % 600 == 0


def test_comms_drop_silences_link_not_plant(rig_comms):
    probe, executor = rig_comms

    # wait_wired keys on the feedback echo, which is (correctly) dead
    # under a total comms fault — arm on the ungated provenance stream
    # instead: it proves the bridge is alive and publishing.
    assert spin_until(
        executor, lambda: len(probe.fault_samples) > 0, timeout_s=10.0
    ), "fault-provenance stream dead -- bridge not spinning?"

    probe.volume_state_pub.publish(Float64(data=SEED_VOLUME_M3))
    probe.valves_pub.publish(UInt8(data=BCU_MOTOR_VALVE_MASK))
    probe.rpm_pub.publish(Int16(data=HELD_RPM))

    # The Gazebo-facing plant path must keep flowing AND integrating —
    # wait out the pump dead time + spin-up before expecting movement.
    assert spin_until(
        executor,
        lambda: len(set(probe.buoyancy_cmds)) > 1,
        timeout_s=PLANT.pump_response_delay_s + 10.0,
    ), (
        "bladder volume never integrated -- comms gate leaked onto the"
        f" plant path? cmds: {sorted(set(probe.buoyancy_cmds))[:3]}"
    )

    # Every bridged telemetry stream is silent.
    assert probe.fb_samples == [], "feedback RPM leaked through the comms gate"
    assert probe.pressure_samples == [], "tank pressure leaked through"
    assert probe.volume_samples == [], "volume telemetry leaked through"
    assert probe.flow_samples == [], "flow rate leaked through"
    assert probe.valve_fb_samples == [], "valve feedback leaked through"
    # ...while the ungated provenance stream kept flowing all along.
    assert len(probe.fault_samples) > 0
