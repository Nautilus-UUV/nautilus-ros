"""Pull recorded streams out of a sweep run's MCAP bag as 1-D arrays.

Time is rebased to seconds since the first sample so multiple runs overlay on a
shared axis. Topics are read post-throttle (1 Hz) with a raw-topic fallback for
older sweeps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

_TYPESTORE = get_typestore(Stores.ROS2_JAZZY)


def _rebase(ts: list[int]) -> np.ndarray:
    t_ns = np.asarray(ts, dtype=np.int64)
    return (t_ns - t_ns[0]).astype(np.float64) * 1e-9


def read_odometry(
    bag_dir: Path, model_name: str = "glider_nautilus"
) -> dict[str, np.ndarray]:
    """Return arrays `t, x, y, z, roll, pitch, yaw` (radians) from the odometry bag.

    Empty arrays if the bag has no odometry — that means "nothing to plot for this
    run" rather than a hard error.
    """
    bag_dir = Path(bag_dir)
    base = f"/model/{model_name}/odometry"
    candidates = (f"{base}/throttled", base)
    ts: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    quat: list[tuple[float, float, float, float]] = []

    with Reader(bag_dir) as reader:
        connections = [c for c in reader.connections if c.topic in candidates]
        if not connections:
            return _empty_odom()
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = _TYPESTORE.deserialize_cdr(rawdata, conn.msgtype)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            ts.append(timestamp)
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
            quat.append((q.x, q.y, q.z, q.w))

    if not ts:
        return _empty_odom()

    euler = Rotation.from_quat(np.asarray(quat)).as_euler("xyz")  # roll, pitch, yaw
    return {
        "t": _rebase(ts),
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "z": np.asarray(zs),
        "roll": euler[:, 0],
        "pitch": euler[:, 1],
        "yaw": euler[:, 2],
    }


def read_fault_levels(bag_dir: Path) -> dict[str, np.ndarray]:
    """Return arrays `t` (s) and `level` (int) from `/bcu/rpm/fault` — the latched
    BCU degradation ladder (0..5). Empty arrays if the topic is absent.
    """
    bag_dir = Path(bag_dir)
    candidates = ("/bcu/rpm/fault/throttled", "/bcu/rpm/fault")
    ts: list[int] = []
    levels: list[int] = []

    with Reader(bag_dir) as reader:
        connections = [c for c in reader.connections if c.topic in candidates]
        if not connections:
            return _empty_fault()
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = _TYPESTORE.deserialize_cdr(rawdata, conn.msgtype)
            ts.append(timestamp)
            levels.append(int(msg.data))

    if not ts:
        return _empty_fault()
    return {"t": _rebase(ts), "level": np.asarray(levels, dtype=np.int64)}


def _empty_odom() -> dict[str, np.ndarray]:
    keys = ("t", "x", "y", "z", "roll", "pitch", "yaw")
    return {k: np.empty(0, dtype=np.float64) for k in keys}


def _empty_fault() -> dict[str, np.ndarray]:
    return {"t": np.empty(0, dtype=np.float64), "level": np.empty(0, dtype=np.int64)}
