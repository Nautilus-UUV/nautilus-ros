# Nautilus ROS System Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture Overview](#3-architecture-overview)
4. [Packages](#4-packages)
   - [py_pkg](#41-py_pkg)
   - [cpp_pkg](#42-cpp_pkg)
5. [Nodes](#5-nodes)
   - [ekf_prefilter](#51-ekf_prefilter-node)
   - [ekf_node](#52-ekf_node)
6. [EKF Algorithm](#6-ekf-algorithm)
   - [State Vector](#61-state-vector)
   - [Prediction Step](#62-prediction-step)
   - [Update Step](#63-update-step)
   - [Quaternion Math](#64-quaternion-math)
7. [Topics and Message Types](#7-topics-and-message-types)
8. [Build System](#8-build-system)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Running the System](#10-running-the-system)
11. [Configuration and Parameters](#11-configuration-and-parameters)
12. [Known Limitations](#12-known-limitations)

---

## 1. Project Overview

**Nautilus ROS** is a ROS 2 (Humble) software stack for the **Nautilus UUV (Unmanned Underwater Vehicle)**. It runs on a Raspberry Pi and provides sensor processing and state estimation capabilities for the vehicle.

The current implementation focuses on an **IMU-based state estimation pipeline** using an Extended Kalman Filter (EKF). The pipeline takes raw IMU measurements, preprocesses them with a smoothing filter, and estimates the vehicle's 3D position, velocity, and orientation.

**Platform:** Raspberry Pi
**ROS Version:** ROS 2 Humble
**Primary Language:** Python (active nodes), C++ (placeholder package)
**Build Tool:** `colcon`

---

## 2. Repository Structure

```
nautilus-ros/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── docs/
│   └── system.md               # This document
├── src/
│   ├── cpp_pkg/                # C++ template package (no nodes yet)
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   └── py_pkg/                 # Python package (active implementation)
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       ├── resource/
│       │   └── py_pkg          # ament package marker
│       ├── py_pkg/             # Python module
│       │   ├── __init__.py
│       │   ├── ekf_filter.py   # Core EKF algorithm
│       │   ├── ekf_node.py     # ROS2 node: runs EKF
│       │   └── ekf_prefilter.py# ROS2 node: IMU smoothing
│       └── test/
│           ├── test_copyright.py
│           ├── test_flake8.py
│           └── test_pep257.py
├── README.md
└── .gitignore
```

---

## 3. Architecture Overview

The system is structured as a two-stage processing pipeline:

```
┌─────────────────┐
│  IMU Hardware   │
│  (or simulator) │
└────────┬────────┘
         │  /imu/left
         │  sensor_msgs/Imu
         ▼
┌─────────────────────────────────────────┐
│           ekf_prefilter node            │
│                                         │
│  Exponential Moving Average filter      │
│  applied to acceleration and angular    │
│  velocity channels independently.       │
│  alpha = 0.5                            │
└────────┬────────────────────────────────┘
         │  /imu/filtered/left
         │  sensor_msgs/Imu
         ▼
┌─────────────────────────────────────────┐
│               ekf_node                  │
│                                         │
│  Extended Kalman Filter                 │
│  State: [pos(3), vel(3), quaternion(4)] │
│  Prediction: accel + gyro               │
│  Update: accel as gravity reference     │
└────────┬────────────────────────────────┘
         │  /position/estimation
         │  geometry_msgs/Pose
         ▼
┌─────────────────┐
│  Downstream     │
│  consumers /    │
│  visualization  │
└─────────────────┘
```

---

## 4. Packages

### 4.1 `py_pkg`

The active implementation package. Contains the full IMU preprocessing and EKF state estimation pipeline.

**Build type:** `ament_python`
**Location:** `src/py_pkg/`

**Manifest (`package.xml`) dependencies:**

| Dependency | Purpose |
|---|---|
| `rclpy` | ROS 2 Python client library |
| `std_msgs` | Standard message types |
| `geometry_msgs` | Point, Twist, Pose messages |
| `sensor_msgs` | Imu message type |

**Entry points (from `setup.py`):**

| Console Script | Module | Entry Function |
|---|---|---|
| `ekf_prefilter` | `py_pkg.ekf_prefilter.ekf_prefilter` | `main` |
| `ekf_node` | `py_pkg.ekf.ekf_node` | `main` |

> **Note:** `package.xml` contains a typo on the `geometry_msgs` dependency (`geomety_msgs`). This may cause dependency resolution issues with some tools. The Python imports work because the packages are installed at the system level regardless.

---

### 4.2 `cpp_pkg`

A skeleton C++ package. It has no executable nodes defined. It exists as a template for future C++ node development.

**Build type:** `ament_cmake`
**Location:** `src/cpp_pkg/`

**CMake dependencies:**

| Dependency | Purpose |
|---|---|
| `rclcpp` | ROS 2 C++ client library |
| `std_msgs` | Standard message types |
| `geometry_msgs` | Geometry message types |
| `sensor_msgs` | Sensor message types |

The `CMakeLists.txt` sets compiler warnings (`-Wall -Wextra -Wpedantic`) and includes the `ament_lint` test suite. No `add_executable()` calls exist yet.

---

## 5. Nodes

### 5.1 `ekf_prefilter` Node

**File:** `src/py_pkg/py_pkg/ekf_prefilter.py`
**Node name:** `ekf_prefilter`

**Purpose:** Conditions raw IMU data before passing it to the EKF. High-frequency sensor noise can cause the EKF to diverge or produce jittery estimates; this stage applies a low-pass filter to smooth the signal.

**Subscriptions:**

| Topic | Type | Description |
|---|---|---|
| `/imu/left` (`UUVTopics.IMU_LEFT`) | `sensor_msgs/Imu` | Raw IMU data from hardware |

**Publications:**

| Topic | Type | Description |
|---|---|---|
| `/imu/filtered/left` (`UUVTopics.IMU_FILTERED_LEFT`) | `sensor_msgs/Imu` | Smoothed IMU data |

**Algorithm — Exponential Moving Average (EMA):**

For each incoming sample, the filter computes:

```
x_filtered = alpha * x_new + (1 - alpha) * x_prev
```

where `alpha = 0.5`. This is applied independently to all 6 IMU channels:
- `linear_acceleration`: x, y, z
- `angular_velocity`: x, y, z

On the very first sample there is no previous value, so the raw measurement is passed through unchanged and stored as the initial filter state.

The published message preserves the input covariance fields. The `orientation_covariance[0]` field is set to `-1.0` to signal downstream nodes that orientation data is not provided.

**Alpha = 0.5 interpretation:**
- Each output is an equal blend of the current measurement and the running average.
- Provides moderate noise rejection with a relatively fast response to real motion changes.
- Increasing alpha toward 1.0 → less smoothing, faster response.
- Decreasing alpha toward 0.0 → more smoothing, slower response.

---

### 5.2 `ekf_node`

**File:** `src/py_pkg/py_pkg/ekf_node.py`
**Node name:** `ekf_node`

**Purpose:** Runs the EKF prediction-update cycle on each incoming filtered IMU message and publishes the estimated position.

**Subscriptions:**

| Topic | Type | Description |
|---|---|---|
| `/imu/filtered/left` (`UUVTopics.IMU_FILTERED_LEFT`) | `sensor_msgs/Imu` | Preprocessed IMU data from `ekf_prefilter` |

**Publications:**

| Topic | Type | Description |
|---|---|---|
| `/position/estimation` (`UUVTopics.POSITION_ESTIMATION`) | `geometry_msgs/Pose` | Estimated 3D position + orientation quaternion |

**ROS 2 Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dt` | `double` | `0.01` | Time step in seconds between EKF updates (assumes 100 Hz IMU) |

**Callback behavior (per IMU message):**

1. Extract `linear_acceleration` (ax, ay, az) as a NumPy array.
2. Extract `angular_velocity` (wx, wy, wz) as a NumPy array.
3. Call `ekf.predict(accel, gyro)` — propagate state forward by `dt`.
4. Call `ekf.update(accel)` — correct attitude estimate using gravity reference.
5. Read position (`ekf.x[0:3]`) and orientation quaternion (`ekf.x[6:10]`) from the EKF state.
6. Publish as `geometry_msgs/Pose` (position + orientation).
7. Log current position to ROS console at INFO level.

> **Note:** Velocity (`ekf.x[3:6]`) is computed inside the EKF but not exposed on any topic.

---

## 6. EKF Algorithm

**File:** `src/py_pkg/py_pkg/ekf_filter.py`

The EKF estimates the vehicle's full kinematic state by fusing IMU measurements. It uses a standard predict/update cycle and represents orientation as a quaternion to avoid gimbal lock.

### 6.1 State Vector

The state is a 10-dimensional vector:

```
x = [ px, py, pz,       <- position (meters, world frame)
      vx, vy, vz,       <- velocity (m/s, world frame)
      qx, qy, qz, qw ]  <- orientation quaternion (body-to-world)
```

**Initial state:**
- Position: origin `[0, 0, 0]`
- Velocity: stationary `[0, 0, 0]`
- Orientation: identity (no rotation) `[0, 0, 0, 1]`

**Covariance matrix P:** 10×10, initialized to identity.

---

### 6.2 Prediction Step

**Called with:** `ekf.predict(accel_body, gyro_body)`
**Inputs:** accelerometer reading in body frame, gyroscope reading in body frame
**Time step:** `dt` (configured via ROS parameter, default 0.01 s)

**Step-by-step:**

1. **Rotate acceleration to world frame:**
   ```
   R = quaternion_to_rotation_matrix(q)
   a_world = R @ a_body
   ```

2. **Compensate for gravity:**
   The accelerometer measures the sum of linear acceleration and the reaction to gravity. Subtracting the known gravitational vector gives the vehicle's true linear acceleration:
   ```
   a_linear = a_world - g          where g = [0, 0, -9.81]
   ```

3. **Integrate kinematics (second-order):**
   ```
   p_new = p + v*dt + 0.5*a_linear*dt²
   v_new = v + a_linear*dt
   ```

4. **Integrate orientation (quaternion kinematics):**
   ```
   q_delta = quaternion_integrate(q, omega, dt)
   q_new = normalize(q * q_delta)
   ```
   where `quaternion_integrate` computes a small rotation quaternion from angular velocity `omega` and time step `dt`.

5. **Covariance propagation:**
   ```
   P = F * P * F^T + Q
   ```
   The state transition Jacobian `F` is a 10×10 matrix that linearizes the nonlinear dynamics around the current state estimate. `Q` is the process noise covariance (tuned constants).

**Process noise covariance Q (diagonal):**

| State components | Noise value |
|---|---|
| Position (px, py, pz) | 0.1 |
| Velocity (vx, vy, vz) | 0.5 |
| Orientation (qx, qy, qz, qw) | 0.01 |

---

### 6.3 Update Step

**Called with:** `ekf.update(measured_accel)`
**Input:** accelerometer reading (in body frame) — used as an attitude reference

**Concept:**
When the vehicle has low translational acceleration, the accelerometer primarily measures gravity. The direction of the measured gravity vector relative to the body frame reveals the vehicle's roll and pitch angles. The EKF uses this as a correction signal.

**Step-by-step:**

1. **Predict expected measurement:**
   The expected accelerometer reading if the current orientation estimate is correct is the gravity vector rotated into body frame:
   ```
   h = -R^T @ g
   ```

2. **Compute innovation (residual):**
   ```
   y = z_measured - h
   ```

3. **Compute Kalman gain:**
   ```
   K = P * H^T * (H * P * H^T + R)^{-1}
   ```
   where `H` is the 3×10 measurement Jacobian (how accelerometer reading changes with state).

4. **State update:**
   ```
   x = x + K * y
   ```

5. **Covariance update:**
   ```
   P = (I - K * H) * P
   ```

6. **Re-normalize quaternion** to preserve unit constraint after the additive correction.

**Measurement noise covariance R:** 3×3 diagonal with value 0.1.

> **Important limitation:** This update step can only correct **roll and pitch** (tilt relative to gravity). It cannot correct **yaw** (heading) because gravity has no horizontal component that would reveal the yaw angle. Yaw estimation requires an additional sensor (magnetometer, GPS, visual odometry, DVL).

---

### 6.4 Quaternion Math

All orientation operations use unit quaternions `q = [qx, qy, qz, qw]` (scalar-last convention).

**Key functions in `ekf_filter.py`:**

| Function | Description |
|---|---|
| `normalize_quaternion(q)` | Divides q by its norm to enforce unit constraint |
| `quaternion_multiply(q1, q2)` | Hamilton product; composes two rotations |
| `quaternion_to_rotation_matrix(q)` | Returns 3×3 rotation matrix from quaternion |
| `quaternion_integrate(q, omega, dt)` | Creates incremental rotation from angular velocity and dt |
| `rotate_vector(q, v)` | Rotates vector v from body to world frame using q |
| `skew(v)` | Converts 3-vector to 3×3 skew-symmetric matrix (used in Jacobians) |

---

## 7. Topics and Message Types

### Active Topics

| Topic | Message Type | Publisher | Subscriber | Description |
|---|---|---|---|---|
| `/imu/left` (`UUVTopics.IMU_LEFT`) | `sensor_msgs/Imu` | Hardware / simulator | `ekf_prefilter` | Raw IMU measurements |
| `/imu/filtered/left` (`UUVTopics.IMU_FILTERED_LEFT`) | `sensor_msgs/Imu` | `ekf_prefilter` | `ekf_node` | EMA-smoothed IMU data |
| `/position/estimation` (`UUVTopics.POSITION_ESTIMATION`) | `geometry_msgs/Pose` | `ekf_node` | Downstream | Estimated pose (position + orientation) |

### Message Structures Used

**`sensor_msgs/Imu`** — fields consumed by this system:
```
geometry_msgs/Vector3 linear_acceleration   # ax, ay, az (m/s^2)
geometry_msgs/Vector3 angular_velocity      # wx, wy, wz (rad/s)
float64[9] linear_acceleration_covariance   # passed through
float64[9] angular_velocity_covariance      # passed through
float64[9] orientation_covariance           # set to -1 at [0] = no orientation
```

**`geometry_msgs/Pose`** — fields published:
```
geometry_msgs/Point      position      # x, y, z (m)
geometry_msgs/Quaternion orientation   # qx, qy, qz, qw (unit quaternion, body→world)
```

---

## 8. Build System

### Prerequisites

- ROS 2 Humble installed at `/opt/ros/humble/`
- Python 3 with `setuptools`
- `colcon` build tool
- `numpy` (used by EKF algorithm)

### Build Steps

```bash
# 1. Source ROS 2 environment
source /opt/ros/humble/setup.bash

# 2. Navigate to workspace root
cd ~/Documents/aris/nautilus-ros

# 3. Build all packages
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo

# 4. Source the workspace overlay
source install/setup.bash
```

The `--symlink-install` flag creates symlinks to Python source files instead of copying them, so edits to Python code take effect immediately without rebuilding.

### Build Outputs

After a successful build, the `install/` directory contains:

```
install/
├── py_pkg/
│   └── lib/py_pkg/
│       ├── ekf_prefilter   <- executable entry point
│       └── ekf_node        <- executable entry point
└── cpp_pkg/
    └── (empty — no executables defined yet)
```

### Package Build Types

| Package | Build System | Tool |
|---|---|---|
| `py_pkg` | `ament_python` | `setuptools` |
| `cpp_pkg` | `ament_cmake` | `CMake 3.8+` |

---

## 9. CI/CD Pipeline

**File:** `.github/workflows/ci.yml`
**Trigger:** Push or pull request to `main`
**Environment:** Docker container — `ros:humble` on `ubuntu:latest`

### Pipeline Steps

1. **Checkout** repository.
2. **Source** ROS 2 Humble environment.
3. **Build** workspace with `colcon build`.
4. **Test** with `colcon test` (runs flake8, pep257, copyright checks).
5. **Smoke test:** Publishes a test message to a topic and verifies receipt.

### Test Suite (`src/py_pkg/test/`)

| Test File | Tool | What it checks |
|---|---|---|
| `test_flake8.py` | `ament_flake8` | PEP 8 code style compliance |
| `test_pep257.py` | `ament_pep257` | Docstring formatting (PEP 257) |
| `test_copyright.py` | `ament_copyright` | License header presence (currently skipped) |

> There are no unit tests for the EKF algorithm itself. The test suite currently only validates code style.

---

## 10. Running the System

### Start the Full Pipeline

Open two terminals (or use a launch file once one is created):

**Terminal 1 — IMU preprocessor:**
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run py_pkg ekf_prefilter
```

**Terminal 2 — EKF state estimator:**
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run py_pkg ekf_node
```

### Useful Inspection Commands

```bash
# List all active topics
ros2 topic list

# Inspect raw IMU data
ros2 topic echo /imu/left

# Inspect filtered IMU data
ros2 topic echo /imu/filtered/left

# Inspect EKF pose output
ros2 topic echo /position/estimation

# Check topic publication rate
ros2 topic hz /position/estimation

# Manually inject a test IMU message
ros2 topic pub /imu/left sensor_msgs/msg/Imu \
  "{linear_acceleration: {x: 0.0, y: 0.0, z: 9.81}, \
    angular_velocity: {x: 0.0, y: 0.0, z: 0.0}}"
```

### Setting the `dt` Parameter

```bash
ros2 run py_pkg ekf_node --ros-args -p dt:=0.02
```

This sets the EKF time step to 20 ms (50 Hz). Match this to the actual IMU publish rate for accurate integration.

---

## 11. Configuration and Parameters

### EKF Filter Parameters (hardcoded in `ekf_filter.py`)

All EKF tuning parameters are currently defined directly in `EKFFilter.__init__()`:

| Parameter | Value | Description |
|---|---|---|
| `dt` | From ROS param (default 0.01 s) | Integration time step |
| `g` | `[0, 0, -9.81]` m/s² | Gravitational acceleration vector |
| `Q[0:3, 0:3]` | `0.1 * I` | Position process noise |
| `Q[3:6, 3:6]` | `0.5 * I` | Velocity process noise |
| `Q[6:10, 6:10]` | `0.01 * I` | Orientation process noise |
| `R` | `0.1 * I` (3×3) | Accelerometer measurement noise |
| `P` | `I` (10×10) | Initial state covariance |

### Prefilter Parameters (hardcoded in `ekf_prefilter.py`)

| Parameter | Value | Description |
|---|---|---|
| `alpha` | `0.5` | EMA smoothing factor (0 = max smoothing, 1 = no smoothing) |

### Tuning Guidance

- **Higher Q** → filter trusts the motion model less, responds faster to measurements, noisier output.
- **Lower Q** → filter trusts the motion model more, smoother output, slower to correct errors.
- **Higher R** → filter trusts sensor measurements less, smoother but slower attitude correction.
- **Lower R** → filter trusts sensor measurements more, faster correction, noisier.
- **alpha (prefilter)** → closer to 1.0 = less smoothing; closer to 0.0 = more smoothing.

---

## 12. Known Limitations

### Yaw (Heading) Estimation

The accelerometer-based update step cannot observe yaw because gravity has no horizontal directionality. Yaw will drift freely over time. To fix this, add one of:
- Magnetometer (compass) for absolute heading
- GPS + velocity-based heading estimation
- Doppler Velocity Log (DVL) — standard for underwater vehicles
- Visual/acoustic odometry

### Assumption of Low Translational Dynamics

The update step assumes the vehicle's translational acceleration is small relative to gravity. During high-speed maneuvers or significant accelerations, the accelerometer will measure both gravity and inertial acceleration, causing incorrect attitude corrections. A more robust approach is to detect and gate out updates during high-dynamics periods.

### No Sensor Bias Estimation

Real IMU sensors have gyroscope bias drift and accelerometer scale/offset errors. These are not modeled or compensated in the current EKF state. Over time, gyro bias will cause orientation errors that compound into position errors. Adding bias states to the EKF state vector is a standard extension.

### Fixed Time Step

The EKF assumes a constant time step `dt`. If IMU messages arrive at variable intervals (e.g., due to OS scheduling jitter on the Raspberry Pi), integration errors will accumulate. A more robust implementation reads the message timestamp and computes `dt` from consecutive timestamps.

### Position Drift

Position is estimated by double-integrating acceleration. Any small errors in gravity compensation or bias accumulate quadratically over time. Without an absolute position reference (GPS, acoustic positioning), the position estimate will drift unboundedly. The current system is useful only for short-duration relative position tracking.

### Velocity Is Not Published

Position and orientation are published as a `geometry_msgs/Pose` on `/position/estimation`, but velocity (`ekf.x[3:6]`) is computed and discarded. Downstream nodes cannot currently access velocity estimates.

### No Launch File

There is no ROS 2 launch file. Both nodes must be started manually in separate terminals. A launch file would allow starting the full pipeline with a single command and could also set parameters declaratively.
