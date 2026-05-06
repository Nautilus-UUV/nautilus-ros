# Copied from divetest files: depth_control_node_ControlSystem.py

import typing

from py_pkg.math_utils import Vector
from py_pkg.utils_controls import PIDController

"""
Glider depth control system.

Single PID on gauge pressure (Pa) -> bladder flow ratio q (1/s).
``Vector.z()`` carries gauge pressure (0 at surface, positive submerged);
positive q = "fill bladder, glider sinks". The wire-level inversion to
motor RPM lives in ``depth_node``.
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
    Single-loop pressure-tracking controller for the glider's BCU.

    State, setpoint, and PID I/O are gauge Pa. Output is bladder flow
    ratio q (1/s); positive q sinks. Finite-difference rate / acceleration
    estimates are kept for logging and post-mortem tuning, but are no
    longer fed to any PID. A single PID on the pressure-error directly
    matches the q -> bladder -> buoyancy -> depth plant (essentially a
    drag-damped double integrator).
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

        # Single PID: gauge Pa -> q (1/s). Derivative-on-measurement
        # (Pa/s rate) and a derivative filter handle the 10 Hz +
        # quantized-pressure noise floor.
        self.pid_pressure = PIDController(**config["pid_pressure"])

        # Glide path setpoints, gauge Pa.
        self.low_pressure_pa: float = config["low_pressure_pa"]
        self.high_pressure_pa: float = config["high_pressure_pa"]
        self.target_pressure_pa: float = self.high_pressure_pa

        # Logging
        self.logger = Logger()

        # Past-position ring kept for offline tuning / log analysis even
        # though the PID no longer consumes the finite-difference
        # estimates directly.
        self.previous_positions: list[Vector] = []
        self.num_past_positions: int = 3
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

        # Maintain the past-position ring purely for logging.
        self.previous_positions.append(position)
        self.previous_times.append(time)
        if len(self.previous_positions) > self.num_past_positions:
            self.previous_positions.pop(0)
            self.previous_times.pop(0)

        velocity_estimate = self.estimate_velocity(
            self.previous_positions, self.previous_times
        )
        acceleration_estimate = self.estimate_acceleration(
            self.previous_positions, self.previous_times
        )
        velocity_for_log = velocity_estimate if velocity_estimate is not None else 0.0
        acceleration_for_log = (
            acceleration_estimate if acceleration_estimate is not None else 0.0
        )

        self.logger.glider_log.append(
            [
                time,
                position.x(),
                position.y(),
                position.z(),
                velocity_for_log,
                acceleration_for_log,
                tank * 10,
            ]
            + other_to_log
        )

        # Z-positive-down: positive error (target > current pressure)
        # produces positive q (fill bladder, sink). Wire-level inversion
        # to negative motor RPM happens in depth_node.
        command = self.pid_pressure.update(
            self.target_pressure_pa, position.z(), time
        )

        self.logger.control_log.append(
            [
                time,
                self.target_pressure_pa,
                command,
            ]
        )

        self.prev_command = command

        return command
