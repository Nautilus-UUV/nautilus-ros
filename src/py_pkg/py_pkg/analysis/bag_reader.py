"""Pull ground-truth pose out of a sweep run's MCAP bag.

`/model/{model}/odometry/throttled` is a `nav_msgs/msg/Odometry` recorded at 1 Hz.
We convert the orientation quaternion to roll/pitch/yaw with scipy and return
everything as 1-D float arrays. Time is rebased to seconds since the first sample
so multiple runs can be overlaid on a shared axis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation


_TYPESTORE = get_typestore(Stores.ROS2_JAZZY)


def _candidate_odom_topics(model_name: str) -> tuple[str, ...]:
    # Older sweeps recorded raw streams; newer sweeps record post-throttle.
    base = f"/model/{model_name}/odometry"
    return (f"{base}/throttled", base)


def read_odometry(
    bag_dir: Path, model_name: str = "glider_nautilus"
) -> dict[str, np.ndarray]:
    """Return arrays `t, x, y, z, roll, pitch, yaw` (radians).

    Returns empty arrays if the bag has no odometry on either the throttled or raw
    topic — older sweeps (`sim_data/smoke/`) only recorded sensors, so an empty
    result means "nothing to plot for this run" rather than a hard error.
    """
    bag_dir = Path(bag_dir)
    candidates = _candidate_odom_topics(model_name)
    ts: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    qx: list[float] = []
    qy: list[float] = []
    qz: list[float] = []
    qw: list[float] = []

    with Reader(bag_dir) as reader:
        connections = [c for c in reader.connections if c.topic in candidates]
        if not connections:
            return _empty()
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = _TYPESTORE.deserialize_cdr(rawdata, conn.msgtype)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            ts.append(timestamp)
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
            qx.append(q.x)
            qy.append(q.y)
            qz.append(q.z)
            qw.append(q.w)

    if not ts:
        return _empty()

    t_ns = np.asarray(ts, dtype=np.int64)
    t = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
    quat = np.stack([qx, qy, qz, qw], axis=1)
    euler = Rotation.from_quat(quat).as_euler("xyz")  # roll, pitch, yaw in radians
    return {
        "t": t,
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "z": np.asarray(zs),
        "roll": euler[:, 0],
        "pitch": euler[:, 1],
        "yaw": euler[:, 2],
    }


def _empty() -> dict[str, np.ndarray]:
    keys = ("t", "x", "y", "z", "roll", "pitch", "yaw")
    return {k: np.empty(0, dtype=np.float64) for k in keys}
