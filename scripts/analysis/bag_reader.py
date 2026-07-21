"""Pull recorded streams out of a sweep run's MCAP bag as 1-D arrays.

Time is rebased to seconds since the first sample so multiple runs overlay on a
shared axis. Topics are read post-throttle (1 Hz) with a raw-topic fallback for
older sweeps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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


def _empty_odom() -> dict[str, np.ndarray]:
    keys = ("t", "x", "y", "z", "roll", "pitch", "yaw")
    return {k: np.empty(0, dtype=np.float64) for k in keys}


def _empty_scalar() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)


def read_scalar_streams(
    bag_dir: Path, topics: Iterable[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`{topic: (t_ns, values)}` for several scalar `.data` streams at once.

    One `Reader` for the whole set: opening a sweep bag streams the entire
    `.mcap.zstd` through a temp file (sweeps record with `compression_mode:
    FILE`), so reading four topics in four calls decompresses the bag four
    times. Keys are the requested topic names, whatever candidate they
    resolved through.

    Each topic reads `<topic>/throttled` (the 1 Hz record path) with a
    raw-topic fallback, mirroring `read_odometry`. Timestamps are ABSOLUTE
    bag nanoseconds — deliberately not rebased, so streams from one run can
    be aligned against each other (each stream's own first sample would
    otherwise define a different epoch). Empty arrays for absent topics.
    """
    topics = list(topics)
    # Both candidate spellings map back to the requested topic name.
    by_candidate = {c: t for t in topics for c in (f"{t}/throttled", t)}
    collected: dict[str, tuple[list[int], list[float]]] = {t: ([], []) for t in topics}
    with Reader(Path(bag_dir)) as reader:
        connections = [c for c in reader.connections if c.topic in by_candidate]
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = _TYPESTORE.deserialize_cdr(rawdata, conn.msgtype)
            ts, values = collected[by_candidate[conn.topic]]
            ts.append(timestamp)
            values.append(float(msg.data))
    return {
        topic: (
            (
                np.asarray(ts, dtype=np.int64),
                np.asarray(values, dtype=np.float64),
            )
            if ts
            else _empty_scalar()
        )
        for topic, (ts, values) in collected.items()
    }


def read_scalar_stream(bag_dir: Path, topic: str) -> tuple[np.ndarray, np.ndarray]:
    """`(t_ns, values)` for one scalar `.data` stream (Int16/Int32/Float32...).

    Single-topic form of `read_scalar_streams`; prefer that one when a
    caller needs several streams from the same bag.
    """
    return read_scalar_streams(bag_dir, [topic])[topic]
