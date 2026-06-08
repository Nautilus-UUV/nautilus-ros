"""Support package for the host-side `plot_sweep.py` tool.

`sweep_loader` discovers runs and drops startup failures, `bag_reader` pulls pose
and fault streams out of each run's MCAP, and `plotting/` holds one plot module per
output. Submodules are imported on demand, so importing the package itself stays
cheap — `rosbags` and `matplotlib` are only pulled in when their submodule is touched.
"""
