"""Support package for the host-side `run_analysis.py` tool.

`sweep_loader` discovers runs, drops startup failures, and reads each run's
anomaly label; `bag_reader` pulls pose streams out of each run's MCAP;
`dataset_stats` rolls them into a markdown summary with a per-anomaly-class
breakdown; and `plotting/` holds one plot module per output. Submodules are
imported on demand, so importing the package itself stays cheap — `rosbags`
and `matplotlib` are only pulled in when their submodule is touched.
"""
