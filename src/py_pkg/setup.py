import os
from glob import glob

from setuptools import find_packages, setup

package_name = "py_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (
            os.path.join("share", package_name, "scenarios", "library"),
            glob("py_pkg/scenarios/library/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="simon",
    maintainer_email="simon@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "pathfinding_node = py_pkg.path.pathfinding:main",
            "imu_prefilter = py_pkg.imu_prefilter.imu_prefilter:main",
            "attitude_node = py_pkg.attitude.attitude_node:main",
            "mqtt_bridge_node = py_pkg.mqtt.mqtt_bridge_node:main",
            "bcu_node = py_pkg.pid.bcu_node:main",
            "acu_node = py_pkg.pid.acu_node:main",
            "stm_com_node = py_pkg.stm_com.stm_com_node:main",
            "can_com_node = py_pkg.stm_com.can_com_node:main",
            "liveness_node = py_pkg.liveness.liveness_node:main",
            "run_watchdog = py_pkg.watchdog.run_watchdog_node:main",
            "bcu_debug_node = py_pkg.debug.bcu_debug_node:main",
            "acu_debug_node = py_pkg.debug.acu_debug_node:main",
            "auto_mission = py_pkg.debug.auto_mission:main",
        ],
    },
)
