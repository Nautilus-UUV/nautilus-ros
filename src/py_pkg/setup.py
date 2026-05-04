from setuptools import find_packages, setup

package_name = "py_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
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
            "bcu_oscillator = py_pkg.integration.bcu_oscillator:main",
            "acu_oscillator = py_pkg.integration.acu_oscillator:main",
            "neutral_buoyancy_test = py_pkg.integration.neutral_buoyancy_test:main",
            "trim_and_buoyancy_test = py_pkg.integration.trim_and_buoyancy_test:main",
            "dive_test_for_ekf = py_pkg.integration.dive_test_for_ekf:main",
        ],
    },
)
