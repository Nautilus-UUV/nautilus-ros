"""Sweep post-processing utilities.

`discover_sweep` walks a `sim_data/{sweep_name}/` directory, `read_odometry`
pulls ground-truth pose out of each run's MCAP, and `plot_sweep` lays the
ensemble out on a 2x3 grid. Submodules are imported on demand; importing
`py_pkg.analysis` itself does not pull in `rosbags` or `matplotlib`, so this
subpackage stays safe to traverse under pytest collection on hosts that don't
have the analysis extras installed.
"""
