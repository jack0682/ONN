"""HAL (Hardware Abstraction Layer) package.

Provides sensor and actuator bridges for hardware-agnostic I/O.
"""

from hal.sensor_bridge import SensorBridge
from hal.actuator_bridge import ActuatorBridge

__all__ = ["SensorBridge", "ActuatorBridge"]
