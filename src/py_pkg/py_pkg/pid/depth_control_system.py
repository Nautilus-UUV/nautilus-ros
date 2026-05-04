# Copied from divetest files: depth_control_node_ControlSystem.py

import typing

from py_pkg.math_utils import Vector
from py_pkg.utils_controls import PIDController

"""
Glider depth control system.

The cascade tracks depth via gauge pressure (Pa) — pressure is the
controlled variable end-to-end, with no metres on the control path.
``Vector.z()`` carries gauge pressure (0 at the surface, positive when
submerged). The output ``q`` is bladder flow ratio (1/s).
"""


class Logger:
    """
    A class for logging glider and control data.
    """

    def __init__(self) -> None:
        """
        Initializes a Logger object.
        """

        self.glider_log: list = []
        self.control_log: list = []


"""
Typedef for the StateMachine class

TODO: This may need to be a full class
"""
State = int
StateGraph = typing.Dict[State, typing.List[State]]
diving = State(0)
surfacing = State(1)


class StateMachine:
    """
    A class representing a state machine.

    TODO: Very limited, rules for state transition are not defined

    Attributes:
        state_graph (dict[State, list[State]]): A dictionary representing the state graph.
        initial_state (State): The initial state of the state machine.
        state (State): The current state of the state machine.
    """

    def __init__(self, state_graph: StateGraph, initial_state: State) -> None:
        """
        Initializes a new instance of the StateMachine class.

        Args:
            state_graph (dict[State, list[State]]): A dictionary representing the state graph.
            initial_state (State): The initial state of the state machine.
        """

        self.state: State = initial_state
        self.state_graph: StateGraph = state_graph

    def next(self) -> None:
        """
        Moves the state machine to the next state.
        """

        self.state = self.state_graph[self.state][0]


class DepthControlSystem:
    """
    Cascaded pressure-tracking controller for the glider's BCU.

    Stages: pressure -> pressure_dot -> pressure_ddot -> q. All internal
    state, setpoints, and PID I/O are in gauge Pa (and its derivatives).

    Attributes:
        state_machine (StateMachine): The state machine for managing the glider's state.
        frequency (int): The control system frequency (Hz).
        period (float): The control system period (s).
        time (float): The current time (s).
        prev_update_time (float): The time of the previous update (s).
        prev_command (float): The previous q command (1/s).
        pid_pressure (PIDController): Outer loop on gauge pressure.
        pid_p_dot (PIDController): Middle loop on pressure rate (Pa/s).
        pid_p_ddot (PIDController): Inner loop on pressure acceleration (Pa/s^2);
            output is the bladder flow ratio q (1/s).
        low_pressure_pa (float): Shallowest setpoint extreme (gauge Pa).
        high_pressure_pa (float): Deepest setpoint extreme (gauge Pa).
        target_pressure_pa (float): Active setpoint (gauge Pa).
        logger (Logger): The logger for logging control system data.
        previous_positions (list): Stores the past positions for finite-difference estimates.
        num_past_positions (int): Number of past positions to keep.
    """

    def __init__(self, config: dict) -> None:
        """
        Initializes a new instance of the ControlSystem class.

        Args:
            config (dict): The configuration parameters for the control system.
        """

        # Initialize state machine
        state_graph: StateGraph = {diving: [surfacing], surfacing: [diving]}
        self.state_machine: StateMachine = StateMachine(state_graph, diving)

        self.frequency: int = config["frequency"]
        self.period: float = 1.0 / self.frequency

        self.time: float = 0.0
        self.prev_update_time: float = self.time
        self.prev_command: float = 0.0

        # Cascading PID controllers (all stages operate in pressure units).
        self.pid_pressure = PIDController(**config["pid_pressure"])
        self.pid_p_dot = PIDController(**config["pid_p_dot"])
        self.pid_p_ddot = PIDController(**config["pid_p_ddot"])

        # Glide path setpoints, gauge Pa.
        self.low_pressure_pa: float = config["low_pressure_pa"]
        self.high_pressure_pa: float = config["high_pressure_pa"]
        self.target_pressure_pa: float = self.high_pressure_pa

        # Logging
        self.logger = Logger()

        # Store past positions.  We need at least 3 for central difference acceleration.
        self.previous_positions: list[Vector] = []
        self.num_past_positions: int = 3  # Number of past positions to store
        self.previous_times: list[float] = []

    def estimate_velocity(
        self, positions: list[Vector], times: list[float]
    ) -> float | None:
        """
        Estimate the rate of change of the controlled variable
        (gauge pressure, Pa/s) by forward difference on the last two
        stored samples. Returns ``None`` if there is insufficient data.
        """
        if len(positions) < 2:
            return None  # Not enough data for velocity estimation

        # Use the last two positions and times
        z1 = positions[-1].z()
        z0 = positions[-2].z()
        t1 = times[-1]
        t0 = times[-2]
        return (z1 - z0) / (t1 - t0)

    def estimate_acceleration(
        self, positions: list[Vector], times: list[float]
    ) -> float | None:
        """
        Estimate the second derivative of the controlled variable
        (gauge pressure, Pa/s^2) via central second difference on the
        last three stored samples. Returns ``None`` if there is
        insufficient data.
        """
        if len(positions) < 3:
            return None  # Not enough data for acceleration estimation

        # Use the last three positions and times for central difference
        z2 = positions[-1].z()
        z1 = positions[-2].z()
        z0 = positions[-3].z()
        t2 = times[-1]
        t1 = times[-2]
        t0 = times[-3]
        return (z2 - 2 * z1 + z0) / ((t2 - t1) * (t1 - t0))

    def calc_acc(
        self,
        position: Vector,
        tank: float,
        time: float,
        other_to_log: list = [],
    ) -> float:
        """
        Calculates the bladder flow command (q, 1/s) for the glider.

        Args:
            position (Vector): position.z() = current gauge pressure (Pa).
            tank (float): The current tank level.
            time (float): The current time (s).

        Returns:
            float: q, the bladder flow ratio command (1/s). Positive
            means "fill bladder, glider sinks".
        """

        self.time = time

        if time < self.prev_update_time + self.period:
            self.logger.glider_log.append(
                [
                    time,
                    position.x(),
                    position.y(),
                    position.z(),
                    0.0,
                    0.0,
                    tank * 10,
                ]
                + other_to_log
            )
            return self.prev_command

        self.prev_update_time = time

        # Store the current position and time
        self.previous_positions.append(position)
        self.previous_times.append(time)

        # Keep only the last 'num_past_positions'
        if len(self.previous_positions) > self.num_past_positions:
            self.previous_positions.pop(0)
            self.previous_times.pop(0)

        # Estimate velocity and acceleration
        velocity_estimate = self.estimate_velocity(
            self.previous_positions, self.previous_times
        )
        acceleration_estimate = self.estimate_acceleration(
            self.previous_positions, self.previous_times
        )

        # First few ticks lack data for finite differences; default to 0.0.
        velocity_for_pid = velocity_estimate if velocity_estimate is not None else 0.0
        acceleration_for_pid = (
            acceleration_estimate if acceleration_estimate is not None else 0.0
        )

        # Log the estimates the PID actually consumes, not dead inputs.
        self.logger.glider_log.append(
            [
                time,
                position.x(),
                position.y(),
                position.z(),
                velocity_for_pid,
                acceleration_for_pid,
                tank * 10,
            ]
            + other_to_log
        )

        # pressure -> pressure_dot -> pressure_ddot -> q
        pid_pressure_output = self.pid_pressure.update(
            self.target_pressure_pa, position.z(), time
        )
        pid_p_dot_output = self.pid_p_dot.update(
            pid_pressure_output, velocity_for_pid, time
        )
        pid_p_ddot_output = self.pid_p_ddot.update(
            pid_p_dot_output, acceleration_for_pid, time
        )

        self.logger.control_log.append(
            [
                time,
                self.target_pressure_pa,
                pid_pressure_output,
                pid_p_dot_output,
                pid_p_ddot_output,
            ]
        )

        # Z-positive-down throughout: target_pressure_pa, position.z(),
        # and the finite-difference rate / acceleration estimates all
        # share the convention, so the cascade's sign is already
        # correct — no negation needed. Positive command = "fill bladder,
        # glider sinks" (the wire-level inversion to motor RPM lives in
        # depth_node).
        command = pid_p_ddot_output
        self.prev_command = command

        return command
