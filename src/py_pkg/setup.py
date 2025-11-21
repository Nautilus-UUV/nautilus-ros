from setuptools import find_packages, setup

package_name = 'py_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='simon',
    maintainer_email='simon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [ 
            'dive_test = py_pkg.simulation.sim_dive_test:main',
            'depth_control_node = py_pkg.depth_control_node:main',
            'sim_dive_acu = py_pkg.simulation.sim_dive_test_acu:main',
            'bcu_controller = py_pkg.bcu_controller_node:main',
        ], #copied from simulations setup.py that was used in polaris-dave-simulation
    },
)
