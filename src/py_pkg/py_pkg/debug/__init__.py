"""Manual-override / bench-debug nodes.

These sit outside the closed-loop stack (estimator -> PID -> BCU/ACU) and
poke the low-level actuator topics directly. Intended for hardware bring-up
and bench testing -- when one of these nodes is active, whichever
controller would normally own that topic is being overridden.
"""
