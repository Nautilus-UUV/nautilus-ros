"""
Description: Motor controller class for all motors.
Background: Copy of divetest motor.py
Changes Needed: Add ACU motors and Profile Position Mode functionality.
"""

import ctypes
from time import sleep


class MotorController:
    """
    A class to represent an EPOS Motor Controller
    """

    epos_lib = None
    key_handle = None
    node_id = 1
    error_code = ctypes.c_ulong()

    def __init__(self, library_path: str):
        """
        Constructor for the MotorController class

        :param library_path: Path to the EPOS library
        :param logger: Logger to use (defaults to the logger for this file)
        """
        self.library_path = library_path

    def initialize_motor(
        self,
        device_name: str = "EPOS4",
        mode: int = 3,
        protocol_stack: str = "MAXON SERIAL V2",
        interface_name: str = "USB",
        port_name: str = "USB0",
        baud_rate: int = 1000000,
        timeout: int = 1000,
        acceleration: int = 1000,
        deceleration: int = 1000,
    ):
        """
        Initialize the motor with specified parameters.

        :param device_name: The device name (default is "EPOS4")
        :param protocol_stack: The protocol stack (default is "MAXON SERIAL V2")
        :param interface_name: The interface name (default is "USB")
        :param port_name: The port name (default is "USB0")
        :param baud_rate: The baud rate (default is 1000000)
        :param timeout: The timeout (default is 1000)
        :param acceleration: The acceleration (default is 1000)
        :param deceleration: The deceleration (default is 1000)
        :return: None

        :raises ConnectionError: If the device connection fails
        :raises RuntimeError: If any of the configuration steps fail
        """
        self.epos_lib = ctypes.CDLL(self.library_path)

        self.vcs_connection = self.epos_lib.VCS_OpenDevice(
            ctypes.create_string_buffer(device_name.encode("utf-8")),
            ctypes.create_string_buffer(protocol_stack.encode("utf-8")),
            ctypes.create_string_buffer(interface_name.encode("utf-8")),
            ctypes.create_string_buffer(port_name.encode("utf-8")),
            ctypes.byref(self.error_code),
        )

        if not self.vcs_connection:
            raise ConnectionError(
                f"Failed to open device. Error Code: {self.error_code.value:#010x}"
            )

        self._set_protocol_stack_settings(baud_rate, timeout)
        self._set_operation_mode(mode)
        self._set_velocity_profile(acceleration, deceleration)
        self._enable_motor()

    def _set_protocol_stack_settings(
        self, baud_rate: int = 1000000, timeout: int = 1000
    ):
        """
        Set the protocol stack settings for the motor.

        :param baud_rate: The baud rate (default is 1000000)
        :param timeout: The timeout (default is 1000)
        :return:
        """
        result = self.epos_lib.VCS_SetProtocolStackSettings(
            self.vcs_connection, baud_rate, timeout, ctypes.byref(self.error_code)
        )

        if not result:
            raise RuntimeError(
                f"Failed to set protocol stack settings. Error Code: {self.error_code.value:#010x}"
            )

    def _set_operation_mode(self, mode: int = 3):
        """
        Set the operation mode for the motor.
        :param mode: The operation mode (default is 3 for velocity mode (BCU)), if operation mode = 1 -> profile position mode (ACU)

        :return:
        """
        profile_mode = ctypes.c_char(mode)
        result = self.epos_lib.VCS_SetOperationMode(
            self.vcs_connection,
            self.node_id,
            profile_mode,
            ctypes.byref(self.error_code),
        )
        if not result:
            raise RuntimeError(
                f"Failed to set operation mode. Error Code: {self.error_code.value:#010x}"
            )

    def _set_velocity_profile(self, acceleration: int = 1000, deceleration: int = 1000):
        """
        Set the velocity profile for the motor.

        :param acceleration: The acceleration (default is 1000)
        :param deceleration: The deceleration (default is 1000)
        :return:
        """

        result = self.epos_lib.VCS_SetVelocityProfile(
            self.vcs_connection,
            self.node_id,
            ctypes.c_ulong(acceleration),
            ctypes.c_ulong(deceleration),
            ctypes.byref(self.error_code),
        )
        if not result:
            raise RuntimeError(
                f"Failed to set velocity profile. Error Code: {self.error_code.value:#010x}"
            )

    def _enable_motor(self):
        """
        Enable the motor.

        :return:
        """

        result = self.epos_lib.VCS_SetEnableState(
            self.vcs_connection, self.node_id, ctypes.byref(self.error_code)
        )
        if not result:
            raise RuntimeError(
                f"Failed to enable motor. Error Code: {self.error_code.value:#010x}"
            )

    def set_velocity(self, target_velocity: int):
        """
        Set the motor velocity in RPM.

        :param target_velocity: The target velocity in RPM
        """
        self._set_velocity(target_velocity)

    def stop_motor(self):
        """
        Stop the motor and halt movement.
        """
        self._set_velocity(0)
        self._halt_movement()

    def _set_velocity(self, velocity: int):
        """
        Set the motor velocity in RPM.

        :param velocity: The target velocity in RPM
        :return:
        """
        result = self.epos_lib.VCS_MoveWithVelocity(
            self.vcs_connection,
            self.node_id,
            ctypes.c_long(velocity),
            ctypes.byref(self.error_code),
        )
        if not result:
            raise RuntimeError(
                f"Failed to set motor velocity. Error Code: {self.error_code.value:#010x}"
            )

    def _halt_movement(self):
        """
        Halt the motor movement.

        :return:
        """
        result = self.epos_lib.VCS_HaltVelocityMovement(
            self.vcs_connection, self.node_id, ctypes.byref(self.error_code)
        )
        if not result:
            raise RuntimeError(
                f"Failed to halt velocity movement. Error Code: {self.error_code.value:#010x}"
            )

    def _close(self):
        """
        Disable the motor and close the connection.
        """
        result = self.epos_lib.VCS_SetDisableState(
            self.vcs_connection, self.node_id, ctypes.byref(self.error_code)
        )

        sleep(1)
        result = self.epos_lib.VCS_CloseDevice(
            self.vcs_connection, ctypes.byref(self.error_code)
        )

    def close_motor(self):
        """
        Close the connection to the motor.
        """
        self._close()

    def __repr__(self):
        """
        String representation of the MotorController class
        """
        return f"MotorController({self.library_path})"

    def __str__(self):
        """
        String representation of the MotorController class
        """
        return f"MotorController with library path {self.library_path}"

    def __del__(self):
        """
        Destructor for the MotorController class
        """
        self._close()
