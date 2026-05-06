"""ROS 2 <-> MQTT bridge.

Runs on the mission laptop side of the tether: forwards ROS telemetry up to
the cloud broker and forwards UI commands back down to the glider's ROS
graph. The MQTT topic tree and payload schema this bridge owns are
documented in ``mqtt_bridge_node.py``.
"""
