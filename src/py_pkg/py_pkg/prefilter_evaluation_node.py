import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


class PrefilterEvaluationNode(Node):
    """
    Evaluation node that verifies the EMA prefilter works as a low-pass filter.

    Compares raw IMU data (/imu/left) against filtered IMU data (/filtered_imu_data)
    by tracking the running variance of each signal. A working low-pass filter should
    produce lower variance than the raw input while preserving the mean.

    Subscribes to:
      /imu/left           (sensor_msgs/Imu) — raw IMU
      /filtered_imu_data  (sensor_msgs/Imu) — EMA-filtered IMU

    Metrics (logged periodically and as final summary on shutdown):
      - Variance of linear acceleration (per axis) for raw vs filtered
      - Variance of angular velocity (per axis) for raw vs filtered
      - Mean of each signal (to verify the mean is preserved)
    """

    LOG_INTERVAL = 100  # log every N raw messages

    def __init__(self):
        super().__init__('prefilter_evaluation_node')

        # Running stats for raw signal: accumulate values for variance computation
        self._raw_accel = [[], [], []]   # x, y, z
        self._raw_gyro  = [[], [], []]

        # Running stats for filtered signal
        self._filt_accel = [[], [], []]
        self._filt_gyro  = [[], [], []]

        self._raw_count = 0

        imu_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, '/imu/left',          self._cb_raw,      imu_qos)
        self.create_subscription(Imu, '/filtered_imu_data', self._cb_filtered, 10)

        self.get_logger().info('Prefilter Evaluation Node started.')

    def _cb_raw(self, msg: Imu):
        a = msg.linear_acceleration
        w = msg.angular_velocity
        self._raw_accel[0].append(a.x)
        self._raw_accel[1].append(a.y)
        self._raw_accel[2].append(a.z)
        self._raw_gyro[0].append(w.x)
        self._raw_gyro[1].append(w.y)
        self._raw_gyro[2].append(w.z)

        self._raw_count += 1
        if self._raw_count % self.LOG_INTERVAL == 0:
            self._log_current_stats()

    def _cb_filtered(self, msg: Imu):
        a = msg.linear_acceleration
        w = msg.angular_velocity
        self._filt_accel[0].append(a.x)
        self._filt_accel[1].append(a.y)
        self._filt_accel[2].append(a.z)
        self._filt_gyro[0].append(w.x)
        self._filt_gyro[1].append(w.y)
        self._filt_gyro[2].append(w.z)

    @staticmethod
    def _variance(values):
        n = len(values)
        if n < 2:
            return float('nan')
        mean = sum(values) / n
        return sum((v - mean) ** 2 for v in values) / (n - 1)

    @staticmethod
    def _mean(values):
        if not values:
            return float('nan')
        return sum(values) / len(values)

    def _log_current_stats(self):
        axes = ['x', 'y', 'z']
        self.get_logger().info(f'--- Prefilter stats after {self._raw_count} raw messages ---')

        for i, ax in enumerate(axes):
            rv = self._variance(self._raw_accel[i])
            fv = self._variance(self._filt_accel[i])
            rm = self._mean(self._raw_accel[i])
            fm = self._mean(self._filt_accel[i])
            reduction = (1.0 - fv / rv) * 100.0 if rv > 0 else float('nan')
            self.get_logger().info(
                f'  accel_{ax}  var: raw={rv:.6f}  filt={fv:.6f}  '
                f'reduction={reduction:.1f}%  |  mean: raw={rm:.4f}  filt={fm:.4f}')

        for i, ax in enumerate(axes):
            rv = self._variance(self._raw_gyro[i])
            fv = self._variance(self._filt_gyro[i])
            rm = self._mean(self._raw_gyro[i])
            fm = self._mean(self._filt_gyro[i])
            reduction = (1.0 - fv / rv) * 100.0 if rv > 0 else float('nan')
            self.get_logger().info(
                f'  gyro_{ax}   var: raw={rv:.6f}  filt={fv:.6f}  '
                f'reduction={reduction:.1f}%  |  mean: raw={rm:.4f}  filt={fm:.4f}')

    def _log_final_summary(self):
        self.get_logger().info('=== Final Prefilter Summary ===')
        self._log_current_stats()


def main(args=None):
    rclpy.init(args=args)
    node = PrefilterEvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._log_final_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
