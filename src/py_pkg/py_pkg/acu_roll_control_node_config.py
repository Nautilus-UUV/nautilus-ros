"""
Description: Configuration parameters for ACU roll control node.
Background: Copied and adapted from divetest files: depth_control_node_config.py
"""

init_acu_control = {}
init_acu_control["target_pitch"] = 0.0
init_acu_control["target_roll"] = 0.0
init_acu_control["pid_pitch"] =      { "kp": 1, "ki": 0, "kd": 0 }
init_acu_control["pid_pitch_rate"] = { "kp": 1, "ki": 0, "kd": 0 }
init_acu_control["pid_roll"] =       { "kp": 1, "ki": 0, "kd": 0 }
init_acu_control["pid_roll_rate"] =  { "kp": 1, "ki": 0, "kd": 0 }
init_acu_control["output_limit_pitch"] = 0.14
init_acu_control["output_limit_roll"] =  60.0 #check units       
init_acu_control["high_pitch"] = 0.07
init_acu_control["low_pitch"] = -0.07
init_acu_control["high_roll"] = 30.0
init_acu_control["low_roll"] = -30.0
init_acu_control["frequency"] = 10

init_acu_control["pid_acu"]["integral_limit"] = 100.0
init_acu_control["pid_acu"]["output_limit"] = 100.0

init_pos = {}
init_pos["x"] = 0.0
init_pos["y"] = 0.0
init_pos["z"] = 0.0
init_vel = {}
init_vel["x"] = 0.0
init_vel["y"] = 0.0
init_vel["z"] = 0.0
init_acc = {}
init_acc["x"] = 0.0
init_acc["y"] = 0.0
init_acc["z"] = 0.0

init_motor = {}
init_motor["min_rpm"] = 1000
init_motor["max_rpm"] = 4000

