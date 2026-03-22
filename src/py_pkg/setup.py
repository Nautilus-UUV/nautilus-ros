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
            "pathfinding_node = py_pkg.pathfinding:main",
            "bcu_oscillator = py_pkg.integration.bcu_oscillator:main",
            "neutral_buoyancy_test = py_pkg.integration.neutral_buoyancy_test:main",
        ],
    },
)
