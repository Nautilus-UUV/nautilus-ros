"""Per-subsystem liveness.

A freshness watchdog over the steady glider-side feedback/sensor streams. The
node publishes one ``diagnostic_msgs/DiagnosticArray`` on ``/status/liveness``
saying which subsystems are reporting; the MQTT bridge forwards it to the
operator UI. The freshness decision lives in a pure, ROS-free class
(``watchdog.py``) so it can be unit-tested directly.
"""
