"""
Author: Lisa Lustenberger
Date: November 2025
Description: Control system for ACU controller node. Simple State machine + P-controllers for roll and pitch axes.
"""

import py_pkg.math_utils as SimMath


class AxisController:
    """
    Generic P-controller + state machine for a single axis (roll or pitch).
    The entire class is in UUV reference frame (not motor frame).
    """

    class State:
        STEADY = 0
        SHIFTING = 1

    def __init__(self, name, Kp, position_tolerance, command_tolerance):
        self.name = name
        self.Kp = Kp
        self.position_tolerance = (
            position_tolerance  # acceptable error for steady state
        )
        self.command_tolerance = (
            command_tolerance  # min delta before sending new command
        )

        self.state = AxisController.State.STEADY
        self.current_pos = 0.0  # sensor reading
        self.target_pos = 0.0  # desired motor position
        self.last_commanded_pos = 0.0  # to prevent unnecessary motor commands

    def update_sensor(self, measured_pos):
        # Update current position from sensor reading, UUV reference frame
        self.current_pos = measured_pos

    def compute_control(self, desired_value):
        """
        Runs the P-controller to generate a new target position.
        """
        error = desired_value - self.current_pos
        correction = self.Kp * error
        new_target = self.current_pos + correction
        return new_target

    def update(self, desired_value):
        """
        Main state machine update.
        Returns a command (float) only when a motor update must be sent.
        Otherwise returns None.
        """

        # Determine target position via P-controller
        self.target_pos = self.compute_control(desired_value)
        error = abs(desired_value - self.current_pos)

        if self.state == AxisController.State.STEADY:
            if error > self.position_tolerance:
                # Need to move
                self.state = AxisController.State.SHIFTING
                self.last_commanded_pos = self.current_pos  # force command on next step
                return self.target_pos

        elif self.state == AxisController.State.SHIFTING:
            if error <= self.position_tolerance:
                # Position reached
                self.state = AxisController.State.STEADY

        # Check if we need to issue a new motor command
        if abs(self.target_pos - self.last_commanded_pos) > self.command_tolerance:
            self.last_commanded_pos = self.target_pos
            return self.target_pos

        return None


class ACUController:
    def __init__(self):
        self.roll = AxisController(
            "roll", Kp=0.5, position_tolerance=1.0, command_tolerance=0.5
        )

        self.pitch = AxisController(
            "pitch", Kp=0.5, position_tolerance=1.0, command_tolerance=0.5
        )

    def uuv_to_motor(self, desired_pitch, desired_roll):
        # convert desired pitch and roll from uuv frame to motor frame
        # angles in degrees
        # FOR NOW: assume fixed angles for pitching

        if desired_pitch > 5:
            desired_pitch_motor = 0.065
        elif desired_pitch < -5:
            desired_pitch_motor = -0.065
        else:
            desired_pitch_motor = 0.0

        desired_roll_motor = SimMath.clamp_mag(desired_roll, -25, 25)

        return desired_pitch_motor, desired_roll_motor

    def update(self, desired_roll, desired_pitch, measured_roll, measured_pitch):

        self.roll.update_sensor(measured_roll)
        self.pitch.update_sensor(measured_pitch)

        roll_cmd_uuv = self.roll.update(desired_roll)
        pitch_cmd_uuv = self.pitch.update(desired_pitch)

        pitch_cmd, roll_cmd = self.uuv_to_motor(pitch_cmd_uuv, roll_cmd_uuv)

        # Only send commands that changed
        commands = {}
        if roll_cmd is not None:
            commands["roll"] = roll_cmd
        if pitch_cmd is not None:
            commands["pitch"] = pitch_cmd

        return commands or None
