from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Test launch file for EKF evaluation against a rosbag.

    Topic remapping:
      /imu/left  →  /imu1   so the prefilter receives the bag's IMU data
                            without changing the production source code.

    Run with:
      ros2 launch py_pkg ekf_test.launch.py
      ros2 bag play <path/to/bag> --remap /imu/left:=/imu/left  (no remap needed on bag side)
    """
    return LaunchDescription([

        # Prefilter: remap the bag topic /imu/left to the expected /imu1
        Node(
            package='py_pkg',
            executable='ekf_prefilter',
            name='ekf_prefilter',
            remappings=[('/imu1', '/imu/left')],
        ),

        # Full EKF (predict + update)
        Node(
            package='py_pkg',
            executable='ekf_node',
            name='ekf_node',
        ),

        # Predict-only EKF (no update step)
        Node(
            package='py_pkg',
            executable='ekf_predict_node',
            name='ekf_predict_node',
        ),

        # EKF evaluation: compares both EKF outputs against ground truth
        Node(
            package='py_pkg',
            executable='ekf_evaluation_node',
            name='ekf_evaluation_node',
        ),

        # Prefilter evaluation: checks EMA smoothing effect
        Node(
            package='py_pkg',
            executable='prefilter_evaluation_node',
            name='prefilter_evaluation_node',
        ),

    ])
