"""Sweep post-processing utilities.

`discover_sweep` walks a `sim_data/{sweep_name}/` directory, `read_odometry` pulls
ground-truth pose out of each run's MCAP, and `plot_sweep` lays the ensemble out on a
2x3 grid (X/Y/Z over roll/pitch/yaw) with the nominal trajectory highlighted.
"""

from .sweep_loader import RunEntry, discover_sweep, read_launch_args
from .bag_reader import read_odometry
from .pose_plot import plot_sweep

__all__ = [
    "RunEntry",
    "discover_sweep",
    "read_launch_args",
    "read_odometry",
    "plot_sweep",
]
