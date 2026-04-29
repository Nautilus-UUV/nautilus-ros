"""
Sensor unit conversions for UUV topics.

These helpers turn raw values from the registry's sensor topics
(e.g. ``UUVTopics.EXTERNAL_PRESSURE``) into physically meaningful
units used by control code.
"""

# Standard atmosphere (Pa) — pressure at the water surface, subtracted
# off the absolute reading from the external pressure sensor.
ATMOSPHERIC_PRESSURE_PA = 101_325.0

# Fresh-water density (kg/m^3). Override for salt water if needed.
WATER_DENSITY_KG_M3 = 1_025.0

# Gravitational acceleration (m/s^2).
GRAVITY_M_S2 = 9.806

# BCU pump volumetric efficiency between 1000 and 3000 RPM.
PUMP_EFFICIENCY = 0.93

# BCU pump volumetric displacement per revolution (m^3 / rev).
VOLUME_PER_REV_M3 = 0.32e-6

SECONDS_PER_MINUTE = 60


def pressure_to_depth(
    pressure_pa: float,
    density: float = WATER_DENSITY_KG_M3,
    atmospheric_pa: float = ATMOSPHERIC_PRESSURE_PA,
) -> float:
    """
    Convert absolute fluid pressure (Pa) to depth (metres, positive downward).

    depth = (P_abs - P_atm) / (rho * g)
    """
    return (pressure_pa - atmospheric_pa) / (density * GRAVITY_M_S2)


def q_to_rpm(q: float, bladder_volume: float) -> float:
    """
    Convert bladder flow-rate ratio (1/s) to motor RPM.

    :param q: bladder flow rate as a fraction of total volume per second (1/s)
    :param bladder_volume: bladder volume (m^3)
    :return: motor speed (RPM)
    """
    flow_rate = q * bladder_volume  # m^3/s
    return SECONDS_PER_MINUTE / (VOLUME_PER_REV_M3 * PUMP_EFFICIENCY) * flow_rate
