# Wire STM32 IMU (accel + gyro) into the control stack and the operator UI

## Context

The STM32 firmware now streams a 6-DOF IMU up the UART: 3 accelerometer axes
(var ids `0x2430`–`0x2432`) and 3 gyroscope axes (`0x2433`–`0x2435`, named
`gyro_accel_*` in the OD — the firmware's "accel" suffix on the gyro words is a
misnomer; they are **angular velocity**). Today `stm_com_node` decodes only
pressure / temperature / leak / RPM / valve frames and silently `debug`-logs any
unknown id, so these six new frames are dropped on the floor. The simulator
already produces a `sensor_msgs/Imu` on `/imu/left`, and the whole downstream
chain (prefilter → EKF → MQTT → frontend store) is built around that topic —
but on hardware nothing publishes it.

Goal: decode the six STM frames, convert raw counts to SI units and a sane body
frame, publish them as `sensor_msgs/Imu` on `UUVTopics.IMU_LEFT` so the existing
pipeline lights up unchanged on hardware, handle the cadence problem the extra
frames introduce, and surface the live values as two new gauge boxes on the
dashboard (angular velocity + translational acceleration), per the mock.

**Key finding — the pipeline already exists end to end.** `IMU_LEFT` is
registered (`Imu`, `SENSOR_STREAM` QoS); `ekf_prefilter` subscribes `/imu/left`
→ publishes `/imu/filtered/left`; the MQTT bridge already forwards
`/imu/filtered/left` → `nautilus/telemetry/imu/left` at 10 Hz; the frontend
telemetry store already binds that topic into `imuLeft` as `Sample<ImuMsg>[]`.
So **no new ROS topics, no `uuv_ros_core` changes, no MQTT bridge changes, no new
store bindings** — we feed the existing `/imu/left` from `stm_com_node` and read
the existing `imuLeft` in the dashboard.

## Decisions

- **Single IMU → `IMU_LEFT` only.** One physical sensor; ignore the left/right
  duality. `IMU_RIGHT` stays sim-only; the `imu_right` liveness row honestly
  reads offline on hardware.
- **Accel full scale ±6 g** (`ACC_RANGE=1`): `Accel_mg = int16/32768 × 1000 ×
  2^(1+1) × 1.5 = int16/32768 × 6000`. **Gyro full scale ±2000 °/s**:
  `Gyro_dps = int16/32768 × 2000`.
- **`sensor_msgs/Imu` stays SI** (m/s², rad/s) per REP-145 so the EKF/prefilter
  keep working. The **gauges display °/s (gyro) and mg (accel)** — conversion is
  presentation-only, done in the dashboard, matching the existing pattern where
  the bridge emits Pa and the UI divides to kPa.

## ROS side — `src/nautilus-ros/src/py_pkg/`

1. **`py_pkg/robot_specs.py`** — new "STM32 IMU" block beside the existing UART
   scales (hardware/firmware contract, like `STM_PRESSURE_LSB_PA` /
   `STM_BCU_RPM_SIGN`). Raw per-LSB scales only; SI conversion lives in
   `physics.py` to avoid a circular import. Constants:
   `STM_ACCEL_FULLSCALE_MG = 6000.0`, `STM_ACCEL_MG_PER_LSB = …/32768`,
   `STM_GYRO_FULLSCALE_DPS = 2000.0`, `STM_GYRO_DPS_PER_LSB = …/32768`, plus the
   sensor→FLU remap `IMU_ACCEL_AXIS_MAP = (("y",+1),("x",+1),("z",+1))` and
   `IMU_GYRO_AXIS_MAP = (("y",+1),("x",-1),("z",+1))`.

2. **`py_pkg/physics.py`** — pure, ROS-free, Tier-1-testable helpers mirroring
   `pressure_to_depth` / `q_to_rpm`:
   - `accel_counts_to_mps2(counts)` = `counts * STM_ACCEL_MG_PER_LSB / 1000 * GRAVITY_M_S2`
   - `gyro_counts_to_rads(counts)` = `counts * STM_GYRO_DPS_PER_LSB * pi/180`
   - `imu_counts_to_body(ax,ay,az,gx,gy,gz)` → SI body-frame `(accel_xyz, gyro_xyz)`
     applying the axis maps. One place the frame convention lives.

3. **`py_pkg/stm_com/stm_com_node.py`** — import `sensor_msgs/Imu`; add var ids
   `0x2430`–`0x2435`; add `self._imu_pub = create_publisher_for_topic(self,
   UUVTopics.IMU_LEFT)`; cache the six latest int16 counts (unpack `<h` with the
   existing length-check pattern). In `_dispatch`, six branches fill the cache;
   on `GYRO_Z_VAR_ID` (`0x2435`, last of the contiguous batch) call
   `_publish_imu()`, which builds the `Imu` (stamp = node clock, frame_id =
   sim IMU link, accel/gyro from `imu_counts_to_body`,
   `orientation_covariance[0] = -1.0`, zeroed accel/gyro covariances).

4. **Critical cadence fix.** `_poll_serial` currently calls `_send_setpoints()`
   after **every** inbound frame (the STM treats each of our frames as a latch
   cue). Six IMU frames per batch (and faster IMU later) would multiply the
   downward RPM/valve TX. Fix: **IMU frames must not cue a setpoint send** —
   `_dispatch` reports whether the frame is a setpoint cue (False for the six IMU
   ids) and `_poll_serial` only sends on a cue, preserving today's RPM/valve
   cadence exactly. Ingest is already burst-safe (drains all bytes / parses all
   frames per tick); for a faster IMU raise the `poll_period_s` param.

5. **Tests.** Tier 1 `test/unit/test_imu_conversions.py` — scale + axis/sign
   assertions (forward→+x, left→+y, up→+z≈+g; roll-right→+ωx, pitch-up→−ωy,
   yaw-left→+ωz; this test is the executable record of the sign table). Tier 2
   extend `test/node/test_stm_com_node.py` — inject the six frames → one `Imu` on
   `/imu/left` with `orientation_covariance[0] == -1.0`; guard that IMU frames
   produce no `0x2102`/`0x2106` TX while a status frame still does.

## Frontend — `nautilus-command-bridge-frontend/`

6. **`src/components/CircularGauge.vue`** — additive `size?: 'normal' |
   'compact'` prop (default `normal`); `compact` shrinks width ~160→~100px + the
   value font so three fit a box. Existing four dials untouched.

7. **`src/views/Dashboard.vue`** — add `imuLeft` to `storeToRefs`, `latestImu`
   computed, two gauge-spec arrays reading `latestImu` with presentation-only
   conversions: angular velocity (Roll/Pitch/Yaw, `rad/s * 180/π`, `°/s`, signed,
   ±180) and acceleration (X/Y/Z, `m/s² / 9.80665 * 1000`, `mg`, signed, ±2000).
   Insert a `.imu-boxes` flex row between `.gauges` and `.stage` with two
   `.data-box` panels (ANG VEL left, TRANS ACC right, `space-between`), each
   three `compact` `CircularGauge`s; scoped CSS following panel conventions.
   Store / MQTT bridge / `ImuMsg` types unchanged.

## Axis / sign validation (bring-up checklist)

The firmware's documented direction/rotation tables aren't a single
right-handed triad, so confirm the remap on hardware before trusting it:
1. **Static, level** → `linear_acceleration ≈ (0, 0, +9.81)`.
2. **Tilt nose-down / roll-right** → gravity shifts onto −x / −y.
3. **Manual rotations**: roll-right → +`angular_velocity.x`; pitch-up →
   −`angular_velocity.y`; yaw-left → +`angular_velocity.z`.
Any mismatch = flip the offending entry in `IMU_ACCEL_AXIS_MAP` /
`IMU_GYRO_AXIS_MAP` and its assertion in `test_imu_conversions.py`.

## Verification

ROS (host build env — `unset CONDA_*`, source ros + install):
```bash
cd src/nautilus-ros/src/py_pkg
/usr/bin/python3 -m pytest test/unit/test_imu_conversions.py -v   # Tier 1
/usr/bin/python3 -m pytest test/node/test_stm_com_node.py -v      # Tier 2
```
Frontend: `cd nautilus-command-bridge-frontend && npm run dev` (:3000); publish a
synthetic `nautilus/telemetry/imu/left` and confirm the two boxes animate —
static IMU shows Accel-Z near +1000 mg and the rest near zero.
