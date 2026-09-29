"""The mission-parameter table, once.

Deliberately a leaf module with no ROS imports. Both a node
(``debug/auto_mission.py``) and a launch file
(``launch/mission_autostart.launch.py``) need this table, and a launch file is
evaluated in a plain Python process that has no business importing rclpy: going
through ``auto_mission`` instead pulled in ``uuv_ros_core``'s message-type
registry and with it the whole of ``sensor_msgs.msg``, ``nav_msgs`` and
``nautilus_msgs`` -- ~180 ms of import, paid on every launch evaluation
(including ``mission_autostart:=false``) and so once per sweep run.

The table drives all four places a mission field used to be spelled out:
``auto_mission``'s ``declare_parameter`` calls, its ``MissionCommand`` fill, its
log line, and the launch file's ``DeclareLaunchArgument`` + ``ParameterValue``
pair. Adding a mission field is an entry here plus the ``.msg``, instead of five
hand-matched edits that nothing checks.

Entries are ``(ROS parameter name, MissionCommand field, type, default)``. The
parameter name is also what the log prints, which is why ``n_oscillations`` can
keep its operator-facing name while the wire field stays ``n_resurfaces``.
"""

MISSION_FIELDS: tuple[tuple[str, str, type, object], ...] = (
    ("mission_id", "mission_id", int, 1),
    ("target_pressure_pa", "target_pressure_pa", float, 0.0),
    ("shallow_pressure_pa", "shallow_pressure_pa", float, 0.0),
    ("angle_rad", "angle_rad", float, 0.0),
    ("n_oscillations", "n_resurfaces", int, 0),
    ("n_steps", "n_steps", int, 1),
)
