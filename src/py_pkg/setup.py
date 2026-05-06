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
            "ekf_prefilter = py_pkg.ekf_prefilter.ekf_prefilter:main",
            "ekf_node = py_pkg.ekf.ekf_node:main",
            "mqtt_bridge_node = py_pkg.mqtt.mqtt_bridge_node:main",
            "depth_node = py_pkg.pid.depth_node:main",
            "acu_node = py_pkg.pid.acu_node:main",
        ],
    },
)
