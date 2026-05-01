# Issues

## Summary

| # | File | Line | Severity | Issue |
|---|---|---|---|---|
| 1 | `ekf_filter.py` | 171 | Critical | Gravity sign wrong (`+ self.g` should be `- self.g`) |
| 2 | `ekf_filter.py` | 238 | Critical | Additive quaternion update — geometrically incorrect |
| 3 | `package.xml` | 12 | Bug | Typo `geomety_msgs` → `geometry_msgs` |
| 4 | `ekf_filter.py` | 196 | Bug | Jacobian uses corrupted `accel_world` from issue #1 |
| 5 | `ekf_node.py` | 16 | Minor | Incomplete docstring |
| 6 | `ekf_filter.py` | 208 | Minor | Missing space in docstring |
| 7 | `ekf_node.py` | 65–68 | Design | Fixed `dt` ignores actual message timing |
| 8 | `ekf_node.py` | 78 | Design | 100 Hz logging floods console |
| 9 | `ekf_node.py` | 65–68 | Design | Same `accel` used for predict and update — double-counting |

---

## Critical

### 1. Gravity compensation is inverted — `ekf_filter.py:171`

```python
accel_world = accel_world + self.g   # self.g = [0, 0, -9.81]
```

An accelerometer at rest measures `+9.81 m/s²` upward (reaction force against gravity). To recover linear acceleration, gravity must be **subtracted**, not added. With the current sign, a stationary vehicle produces `a_linear = [0, 0, -9.81]`, causing the filter to believe the vehicle is in free-fall and driving unbounded downward velocity and position drift.

**Fix:**
```python
accel_world = accel_world - self.g
```

---

### 2. Additive quaternion update corrupts orientation — `ekf_filter.py:238`

```python
self.x = self.x + K @ y
self.x[6:10] = normalize_quaternion(self.x[6:10])
```

`K @ y` produces a 10-vector that is added directly to the quaternion components. Quaternions live on a non-Euclidean manifold (the unit 3-sphere); adding an arbitrary delta and renormalizing is not a valid rotation composition except for infinitesimally small corrections. For any meaningful innovation this produces a geometrically incorrect orientation that will cause the filter to diverge.

**Fix:** Use a multiplicative update — convert the 3D orientation correction `(K @ y)[6:9]` into a small rotation quaternion and compose it with `q` via `quaternion_multiply`.

---

## Bugs

### 3. Typo in `package.xml` breaks dependency resolution — `package.xml:12`

```xml
<depend>geomety_msgs</depend>
```

`geomety_msgs` is not a valid ROS package name. `rosdep` and `colcon` dependency checks will fail to resolve it.

**Fix:** Correct to `geometry_msgs`.

---

### 4. State transition Jacobian uses corrupted acceleration — `ekf_filter.py:196`

```python
F[3:6, 6:9] = -skew(accel_world) * dt
```

`accel_world` at this point already has gravity added with the wrong sign (issue #1), so the Jacobian is computed from an incorrect vector. This causes the covariance to propagate incorrectly even if issue #1 is fixed independently.

**Fix:** Resolve issue #1 first; this issue is then corrected automatically since both use the same variable.

---

## Minor

### 5. Incomplete docstring — `ekf_node.py:16`

```python
"""
ROS2 Node ...
Subscribes to: /filtered_imu_data (sensor_msgs/Imu)
Publishes to:
"""
```

The `Publishes to:` line is empty.

**Fix:** Add `/ekf_position (geometry_msgs/Point)`.

---

### 6. Broken word in docstring — `ekf_filter.py:208`

```python
"""...underthe assumption that...
```

Missing space between `under` and `the`.

**Fix:** `under the assumption that`.

---

## Design Issues

### 7. Fixed `dt` ignores actual message timing — `ekf_node.py:65–68`

The EKF integrates with a hardcoded `dt=0.01 s`. On a Raspberry Pi, OS scheduling jitter means IMU messages rarely arrive at exactly 100 Hz. Any deviation between the assumed and actual interval introduces integration error that accumulates silently over time.

**Fix:** Compute `dt` dynamically from consecutive message timestamps:
```python
dt = (msg_in.header.stamp - self.last_stamp).nanoseconds * 1e-9
self.last_stamp = msg_in.header.stamp
```

---

### 8. Per-callback logging at full IMU rate — `ekf_node.py:78`

```python
self.get_logger().info(f'Current position ...')
```

At 100 Hz this emits 100 log lines per second, flooding the console and adding non-trivial CPU overhead on the Raspberry Pi.

**Fix:** Rate-limit logging, e.g. log only every N callbacks or use a separate timer publishing at 1–2 Hz.

---

### 9. Same accelerometer sample used for both predict and update — `ekf_node.py:65–68`

```python
self.ekf.predict(measured_accel, measured_gyro)
self.ekf.update(measured_accel)
```

The identical `measured_accel` is passed to both steps. The predict step has already integrated this acceleration into the state; passing it again to the update step as an independent gravity measurement double-counts the same data, biasing the attitude correction.

**Fix:** Either use gyro-only prediction (no accelerometer in predict) and accel-only update, or accept the double-counting and retune `R` accordingly. The cleanest separation is gyro → predict, accel → update.
