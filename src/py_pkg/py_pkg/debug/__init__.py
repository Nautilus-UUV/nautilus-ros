"""Manual-override / bench-debug nodes.

These sit outside the closed-loop stack (estimator -> PID -> BCU/ACU) and
poke the low-level actuator topics directly. Intended for hardware bring-up
and bench testing -- when one of these nodes is active, whichever
controller would normally own that topic is being overridden.
"""

# Held commands re-publish at 10 Hz because the MQTT egress is a
# rate-limit-and-drop throttle (EGRESS_MAP caps periodic telemetry at
# 10 Hz); each tick refreshes the bridge's per-topic clock so a held
# value can't lose the race against the throttle. Shared by every debug
# node so a change to the tether throttle is a one-place edit here.
PUBLISH_PERIOD_S = 0.1

# Ticks to keep carrying the terminal 0/neutral after a command ends
# before going silent. 5 @ 10 Hz = 0.5 s, comfortably above the egress
# throttle, so the reset-to-0 reliably lands on the UI.
FLUSH_TICKS = 5
