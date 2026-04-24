import ctypes
from time import sleep


class MotorControllerPPM:
    """EPOS4 motor controller using Profile Position Mode (via Maxon VCS API).

    This class wraps the EPOS VCS C API (loaded with ctypes) and provides
    convenience methods to configure the position profile and command
    profile-position moves.

    Notes:
    - The underlying VCS library must be available and its C symbols
      match the names used here (typical for Maxon VCS DLL/so).
    - Methods raise RuntimeError on VCS call failures and ConnectionError
      when opening the device fails.
    """

    def __init__(self, library_path: str, node_id: int = 1):
        self.library_path = library_path
        self.node_id = node_id
        self.epos_lib = None
        self.vcs_connection = None
        self.error_code = ctypes.c_ulong()

    def open(self,
             device_name: str = "EPOS4",
             protocol_stack: str = "MAXON SERIAL V2",
             interface_name: str = "USB",
             port_name: str = "USB0",
             baud_rate: int = 1000000,
             timeout: int = 1000):
        """Open the device and set protocol stack settings."""
        self.epos_lib = ctypes.CDLL(self.library_path)

        self.vcs_connection = self.epos_lib.VCS_OpenDevice(
            ctypes.create_string_buffer(device_name.encode("utf-8")),
            ctypes.create_string_buffer(protocol_stack.encode("utf-8")),
            ctypes.create_string_buffer(interface_name.encode("utf-8")),
            ctypes.create_string_buffer(port_name.encode("utf-8")),
            ctypes.byref(self.error_code),
        )

        if not self.vcs_connection:
            raise ConnectionError(f"Failed to open device. Error Code: {self.error_code.value:#010x}")

        result = self.epos_lib.VCS_SetProtocolStackSettings(
            self.vcs_connection, ctypes.c_ulong(baud_rate), ctypes.c_ulong(timeout), ctypes.byref(self.error_code)
        )
        if not result:
            raise RuntimeError(f"Failed to set protocol stack settings. Error Code: {self.error_code.value:#010x}")

    def configure_profile_position(self, max_velocity: int, acceleration: int, deceleration: int):
        """Set operation mode to Profile Position and configure position profile.

        :param max_velocity: maximum profile velocity
        :param acceleration: profile acceleration
        :param deceleration: profile deceleration
        """
        # Operation mode 1 == Profile Position Mode
        profile_position_mode = ctypes.c_char(1)
        result = self.epos_lib.VCS_SetOperationMode(
            self.vcs_connection, ctypes.c_ushort(self.node_id), profile_position_mode, ctypes.byref(self.error_code)
        )
        if not result:
            raise RuntimeError(f"Failed to set operation mode. Error Code: {self.error_code.value:#010x}")

        # Configure position profile (VCS_SetPositionProfile(handle, nodeId, vel, acc, dec, &err))
        result = self.epos_lib.VCS_SetPositionProfile(
            self.vcs_connection,
            ctypes.c_ushort(self.node_id),
            ctypes.c_ulong(max_velocity),
            ctypes.c_ulong(acceleration),
            ctypes.c_ulong(deceleration),
            ctypes.byref(self.error_code),
        )
        if not result:
            raise RuntimeError(f"Failed to set position profile. Error Code: {self.error_code.value:#010x}")

        # Enable controller (switch on / enable operation)
        result = self.epos_lib.VCS_SetEnableState(self.vcs_connection, ctypes.c_ushort(self.node_id), ctypes.byref(self.error_code))
        if not result:
            raise RuntimeError(f"Failed to enable motor. Error Code: {self.error_code.value:#010x}")

    def move_to_position(self, position: int, absolute: bool = True, immediately: bool = True):
        """Move to the specified target position (in device units).

        :param position: target position
        :param absolute: absolute (True) or relative (False)
        :param immediately: start immediately if True
        """
        abs_flag = ctypes.c_bool(bool(absolute))
        imm_flag = ctypes.c_bool(bool(immediately))
        result = self.epos_lib.VCS_MoveToPosition(
            self.vcs_connection,
            ctypes.c_ushort(self.node_id),
            ctypes.c_long(position),
            abs_flag,
            imm_flag,
            ctypes.byref(self.error_code),
        )
        if not result:
            raise RuntimeError(f"Failed to move to position. Error Code: {self.error_code.value:#010x}")

    def halt_position(self):
        """Halt ongoing position movement."""
        result = self.epos_lib.VCS_HaltPositionMovement(self.vcs_connection, ctypes.c_ushort(self.node_id), ctypes.byref(self.error_code))
        if not result:
            raise RuntimeError(f"Failed to halt position movement. Error Code: {self.error_code.value:#010x}")

    def disable(self):
        """Disable the drive (set disable state)."""
        result = self.epos_lib.VCS_SetDisableState(self.vcs_connection, ctypes.c_ushort(self.node_id), ctypes.byref(self.error_code))
        if not result:
            raise RuntimeError(f"Failed to disable motor. Error Code: {self.error_code.value:#010x}")

    def close(self):
        """Close the device connection and cleanup."""
        try:
            if self.epos_lib and self.vcs_connection:
                # try disabling first
                try:
                    self.disable()
                except Exception:
                    pass

                sleep(0.1)
                self.epos_lib.VCS_CloseDevice(self.vcs_connection, ctypes.byref(self.error_code))
        finally:
            self.vcs_connection = None
            self.epos_lib = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
