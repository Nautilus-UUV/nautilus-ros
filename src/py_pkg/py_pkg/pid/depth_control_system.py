# Copied from divetest files: depth_control_node_ControlSystem.py

import typing

from py_pkg.math_utils import Vector
from py_pkg.utils_controls import PIDController

"""
This is the Glider's control system module.

It provides a state machine class and logging. The PID controller lives in
py_pkg.utils_controls.
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
    The ControlSystem class represents the control system for the glider simulation.

    Attributes:
        state_machine (StateMachine): The state machine for managing the glider's state.
        frequency (int): The control system frequency.
        period (float): The control system period.
        time (float): The current time.
        prev_update_time (float): The time of the previous update.
        prev_command (float): The previous command value.
        pid_depth (PIDController): The PID controller for depth control.
        pid_v_vel (PIDController): The PID controller for vertical velocity control.
        pid_v_acc (PIDController): The PID controller for vertical acceleration control.
        min_depth (float): The minimum depth for the glider.
        max_depth (float): The maximum depth for the glider.
        target_depth (float): The target depth for the glider.
        logger (Logger): The logger for logging control system data.
        previous_positions (list): Stores the past positions for calculating derivatives
        num_past_positions (int): The number of past positions to use for calculations
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

        # Create cascading PID controllers
        self.pid_depth = PIDController(**config["pid_depth"])
        self.pid_v_vel = PIDController(**config["pid_v_vel"])
        self.pid_v_acc = PIDController(**config["pid_v_acc"])

        # Glide path parameters
        self.min_depth: float = config["high_depth"]
        self.max_depth: float = config["low_depth"]
        self.target_depth: float = self.max_depth

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
        Estimates the vertical velocity using the central difference method.

        Args:
            positions (list[Vector]): A list of past position vectors.
            times (list[float]): A list of corresponding times.

        Returns:
            float | None: The estimated vertical velocity, or None if insufficient data.
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
        Estimates the vertical acceleration using the central difference method.

        Args:
            positions (list[Vector]): A list of past position vectors.
            times (list[float]): A list of corresponding times.

        Returns:
            float | None: The estimated vertical acceleration, or None if insufficient data.
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
        Calculates the acceleration command for the glider.

        Args:
            position (Vector): The current position of the glider.
            tank (float): The current tank level.
            time (float): The current time.

        Returns:
            float: The acceleration command for the glider.
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

        # REMOVED logic since we are manually setting target_depth
        # if self.state_machine.state == diving:
        #     if position.z() <= self.target_depth:
        #         self.target_depth = self.min_depth
        #         self.state_machine.next()
        # else:
        #     if position.z() >= self.target_depth:
        #         self.target_depth = self.max_depth
        #         self.state_machine.next()

        # depth -> v_vel -> v_acc
        pid_depth_output = self.pid_depth.update(self.target_depth, position.z(), time)
        pid_v_vel_output = self.pid_v_vel.update(
            pid_depth_output, velocity_for_pid, time
        )
        pid_v_acc_output = self.pid_v_acc.update(
            pid_v_vel_output, acceleration_for_pid, time
        )

        self.logger.control_log.append(
            [
                time,
                self.target_depth,
                pid_depth_output,
                pid_v_vel_output,
                pid_v_acc_output,
            ]
        )

        command = -pid_v_acc_output
        self.prev_command = command

        return command
