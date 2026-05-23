# Plan: `can_com_node` — ROS 2 → CAN (SocketCAN) actuator bridge

## Context

The Nautilus glider's control stack already ships a UART bridge,
`stm_com_node.py`, that forwards the BCU motor RPM setpoint to the STM32 over
serial. On the real vehicle, BCU/ACU commands are also (or instead) delivered
to a **Control Unit (CU) board over the CAN bus** — the operator's hardware
workflow brings up SocketCAN (`ip link set can0 type can bitrate 125000`,
`cansend`, `candump`).

`can_com_node` subscribes to the four actuator setpoints the control stack
publishes and packs them into a **single 8-byte PDO frame (CAN ID 0x181)**
transmitted on `can0`. It is production-shaped for real deployment, but the
immediate acceptance criterion is the operator's five test cases: BCU RPM =
+100 / −100 / 0, and valve open / close, driven from the frontend (which
already reaches `/bcu/rpm` and `/bcu/valves` via the MQTT bridge →
`bcu_debug_node`).

Decisions confirmed with the user:
- **Valve bits forwarded verbatim**: byte 6 = `/bcu/valves` unchanged
  (bit0 = valve 1 → 0x01, bit1 = valve 2 → 0x02). Matches the field spec,
  `uuv_ros_core`, and the frontend. (The "Open Valve B = 0x01" example
  therefore corresponds to valve 1 / bit 0.)
- **Periodic transmit at 1 Hz, no watchdog**: heartbeat the latest cached
  values once per second; last setpoint latches until changed. No
  auto-zeroing on stale streams.
- **Transport = raw `AF_CAN` socket (Python stdlib)**: no new dependency, no
  python-can, no `cansend` subprocess. The node opens its own `CAN_RAW` socket
  on `can0` — the same syscalls `cansend`/python-can use internally. Linux/
  SocketCAN-only, which is exactly the deployment. (`cansend` stays the
  operator's manual bench tool.)

## CAN frame contract

Single PDO, CAN ID **0x181**, standard 11-bit ID, 8 data bytes, little-endian:

| bytes | field      | type    | units / meaning                          |
|-------|------------|---------|------------------------------------------|
| 0–1   | ACU_pitch  | uint16  | mm of prismatic travel                   |
| 2–3   | BCU        | int16   | rpm, directional (− = deflate/sink)      |
| 4–5   | ACU_roll   | int16   | centidegrees (±3000 = ±30°)              |
| 6     | Valves     | uint8   | bitmask: bit0=valve1, bit1=valve2 (1=open)|
| 7     | reserved   | 0       | spare for future BCU state info          |

Packing: `struct.pack("<HhhBB", pitch_mm, bcu_rpm, acu_roll_cdeg, valves, 0)`,
then wrapped in the kernel `can_frame` (`struct.pack("<IB3x8s", can_id, dlc,
payload)`).

Reproduces the verified example frames exactly:
- rpm=+100 → `00 00 64 00 00 00 00 00`
- rpm=−100 → `00 00 9C FF 00 00 00 00`
- rpm=0    → `00 00 00 00 00 00 00 00`
- valve1 open → `00 00 00 00 00 00 01 00`

The bus **bitrate is owned by the OS** (`ip link`), not the node — it just
attaches to the already-up interface.

## Files changed

1. **New:** `src/py_pkg/py_pkg/stm_com/can_com_node.py` — the node, next to
   `stm_com_node.py`.
2. **Edit:** `src/py_pkg/setup.py` — `console_scripts` entry
   `"can_com_node = py_pkg.stm_com.can_com_node:main"`.
3. **Edit:** `src/py_pkg/launch/control_stack.launch.py` — launch-gated `Node`
   + `enable_can_com` arg (default false) + docstring composition line,
   mirroring the existing `enable_stm_com` pattern.

No `package.xml` change (raw `AF_CAN` socket is stdlib). No `uuv_ros_core`
edits (`BCU_RPM`, `BCU_VALVES`, `ACU_PITCH`, `ACU_ROLL` and their Int16/UInt8 +
CONTROL QoS already exist).

## Node design (`can_com_node.py`)

Mirrors `stm_com_node.py`: module docstring with the wire format, `struct`
constants at module scope, `__init__` = declare-params → open-socket →
subscriptions → tx timer, cache-only callbacks, one `_send_pdo()`,
`destroy_node()` closes the socket, standard `main()`.

- **Subscriptions** (via `create_subscription_for_topic`): `BCU_RPM`,
  `BCU_VALVES`, `ACU_PITCH`, `ACU_ROLL`. Callbacks only cache. Defaults all 0
  (motor off, valves closed, neutral) until first message.
- **Parameters**: `channel` (`"can0"`), `can_id` (`0x181`), `tx_period_s`
  (`1.0` → 1 Hz). No watchdog param (per user choice).
- **Socket**: `socket.socket(AF_CAN, SOCK_RAW, CAN_RAW)` then `bind((channel,))`
  eagerly in `__init__`; fail-fast if `can0` is down (bind raises `OSError`).
  The node is launch-gated so sim/dev hosts never reach it.
- **`_send_pdo()`** (1 Hz timer): clamp pitch to `[0, 0xFFFF]`, pack the PDO,
  wrap in the 16-byte `can_frame`, `sock.send()` guarded by `try/except
  OSError` (warn, don't tear down the timer).
- **`destroy_node()`**: `sock.close()` in a `try/except`.
- Pure hardware **sink** (no `/can/out` echo) — symmetric with `stm_com`;
  `candump can0` gives full bus observability. A `publish_echo` param to light
  up the unused `CAN_OUT` topic is a clean future toggle, out of scope now.

## Edge cases / risks

- **OS owns the bitrate.** Vehicle prerequisite: `ip link set can0 type can
  bitrate 125000 && ip link set up can0` before `enable_can_com:=true`.
- **Negative ACU_PITCH is untested vs firmware.** Topic is signed Int16; CAN
  field is uint16, and `struct.pack("<H", neg)` would raise. Clamp-to-0 is the
  safe minimal handling (correct for the tested range, pitch ≥ 0) but encodes
  negative commanded pitch as 0 mm. The negative-pitch encoding (abs vs offset
  vs unsupported) is a firmware-contract decision to settle before relying on
  negative travel. Do **not** silently `abs()` (flips direction).
- **No watchdog (per user choice):** if the controller wedges, the last RPM/
  valve setpoint keeps being heartbeat. Safety must come from the CU board or
  the upstream emergency-surface path, not this bridge.
- **`sock.send()` can raise `OSError`** (tx queue full / `ENOBUFS`, or
  `ENETDOWN` if `can0` drops); caught so one hiccup doesn't kill the timer.
- **Linux-only.** `socket.AF_CAN`/`CAN_RAW` are stdlib but Linux-only;
  irrelevant here (deployment + dev are Linux).
- **Single producer of ID 0x181** — keep this node the sole transmitter.
- **Naming:** `stm_com/` now hosts a non-STM node. Shipped as-is for minimal
  diff; `stm_com/` → `comms/` rename is future cleanup.

## Verification

Build (no extra deps — raw socket is stdlib):
```bash
cd /home/girji/dave_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select py_pkg && source install/setup.bash
```

No-hardware test on a virtual CAN interface:
```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

candump vcan0 &                                            # watch the bus
ros2 run py_pkg can_com_node --ros-args -p channel:=vcan0  # 1 Hz frames

ros2 topic pub -1 /bcu/rpm    std_msgs/msg/Int16 "{data: 100}"   # → 181#0000640000000000
ros2 topic pub -1 /bcu/rpm    std_msgs/msg/Int16 "{data: -100}"  # → 181#00009CFF00000000
ros2 topic pub -1 /bcu/rpm    std_msgs/msg/Int16 "{data: 0}"     # → 181#0000000000000000
ros2 topic pub -1 /bcu/valves std_msgs/msg/UInt8 "{data: 1}"     # → 181#0000000000000100 (valve1)
ros2 topic pub -1 /bcu/valves std_msgs/msg/UInt8 "{data: 0}"     # → close
```
Confirm `candump` shows ID `181`, 8 bytes, the expected payloads, repeating at
1 Hz. End-to-end from the frontend: bring up the MQTT bridge + `bcu_debug_node`,
toggle manual override, send RPM/valve from the UI, and watch the same frames
on `candump can0` on the vehicle.
