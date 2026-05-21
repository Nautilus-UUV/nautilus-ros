"""ROS 2 <-> STM32 serial bridge.

Forwards the depth PID's BCU RPM setpoint down to the STM32 over a UART
link. The wire protocol is a fixed-frame format owned by the STM firmware
(0xAA sync + 2-byte big-endian variable id + 1-byte length + LE payload);
this package keeps the framing in one place and exposes a single node.
"""
