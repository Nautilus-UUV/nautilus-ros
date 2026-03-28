import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, PoseArray


def position_error(p_est, p_gt):
    """Euclidean distance between estimated and ground truth position."""
    return math.sqrt(
        (p_est[0] - p_gt[0]) ** 2 +
        (p_est[1] - p_gt[1]) ** 2 +
        (p_est[2] - p_gt[2]) ** 2
    )


def orientation_error_deg(q_est, q_gt):
    """Angle in degrees between two quaternions [x, y, z, w].

    Uses the dot product formula: angle = 2 * arccos(|q1 · q2|).
    The absolute value handles the quaternion double-cover (q and -q are the same rotation).
    """
    dot = abs(
        q_est[0] * q_gt[0] +
        q_est[1] * q_gt[1] +
        q_est[2] * q_gt[2] +
        q_est[3] * q_gt[3]
    )
    dot = min(1.0, dot)  # clamp to avoid numerical issues in arccos
    return math.degrees(2.0 * math.acos(dot))


class EkfEvaluationNode(Node):
    """
    Evaluation node that compares the full EKF and the predict-only EKF against
    ground truth from the simulation.

    Subscribes to:
      /ekf_position              (geometry_msgs/Point)      — full EKF position
      /ekf_orientation           (geometry_msgs/Quaternion) — full EKF orientation
      /ekf_predict_position      (geometry_msgs/Point)      — predict-only position
      /ekf_predict_orientation   (geometry_msgs/Quaternion) — predict-only orientation
      /model/glider_nautilus/pose (geometry_msgs/PoseArray) — simulation ground truth

    Metrics (logged on every ground truth message and as final RMSE on shutdown):
      - Position error: Euclidean distance [m]
      - Orientation error: angle between quaternions [degrees]
    """

    def __init__(self):
        super().__init__('ekf_evaluation_node')

        # Latest estimates — updated whenever a new message arrives
        self._ekf_pos = None
        self._ekf_ori = None
        self._pred_pos = None
        self._pred_ori = None

        # Accumulated squared errors for RMSE computation
        self._ekf_pos_sq_errors = []
        self._ekf_ori_sq_errors = []
        self._pred_pos_sq_errors = []
        self._pred_ori_sq_errors = []

        # Subscribers
        self.create_subscription(Point,      '/ekf_position',            self._cb_ekf_pos, 10)
        self.create_subscription(Quaternion, '/ekf_orientation',         self._cb_ekf_ori, 10)
        self.create_subscription(Point,      '/ekf_predict_position',    self._cb_pred_pos, 10)
        self.create_subscription(Quaternion, '/ekf_predict_orientation', self._cb_pred_ori, 10)
        self.create_subscription(PoseArray,  '/model/glider_nautilus/pose', self._cb_ground_truth, 10)

        self.get_logger().info('EKF Evaluation Node started.')

    # --- Estimate callbacks (just store the latest value) ---

    def _cb_ekf_pos(self, msg: Point):
        self._ekf_pos = (msg.x, msg.y, msg.z)

    def _cb_ekf_ori(self, msg: Quaternion):
        self._ekf_ori = (msg.x, msg.y, msg.z, msg.w)

    def _cb_pred_pos(self, msg: Point):
        self._pred_pos = (msg.x, msg.y, msg.z)

    def _cb_pred_ori(self, msg: Quaternion):
        self._pred_ori = (msg.x, msg.y, msg.z, msg.w)

    # --- Ground truth callback (compute and log errors) ---

    def _cb_ground_truth(self, msg: PoseArray):
        if not msg.poses:
            return

        gt_pose = msg.poses[0]
        gt_pos = (gt_pose.position.x, gt_pose.position.y, gt_pose.position.z)
        gt_ori = (gt_pose.orientation.x, gt_pose.orientation.y,
                  gt_pose.orientation.z, gt_pose.orientation.w)

        # Full EKF errors
        if self._ekf_pos is not None and self._ekf_ori is not None:
            ekf_pe = position_error(self._ekf_pos, gt_pos)
            ekf_oe = orientation_error_deg(self._ekf_ori, gt_ori)
            self._ekf_pos_sq_errors.append(ekf_pe ** 2)
            self._ekf_ori_sq_errors.append(ekf_oe ** 2)
        else:
            ekf_pe = ekf_oe = float('nan')

        # Predict-only errors
        if self._pred_pos is not None and self._pred_ori is not None:
            pred_pe = position_error(self._pred_pos, gt_pos)
            pred_oe = orientation_error_deg(self._pred_ori, gt_ori)
            self._pred_pos_sq_errors.append(pred_pe ** 2)
            self._pred_ori_sq_errors.append(pred_oe ** 2)
        else:
            pred_pe = pred_oe = float('nan')

        self.get_logger().info(
            f'pos_err — EKF: {ekf_pe:.3f} m  |  predict-only: {pred_pe:.3f} m  ||  '
            f'ori_err — EKF: {ekf_oe:.2f}°  |  predict-only: {pred_oe:.2f}°'
        )

    def _log_final_rmse(self):
        """Compute and log final RMSE for all metrics."""
        def rmse(sq_errors):
            if not sq_errors:
                return float('nan')
            return math.sqrt(sum(sq_errors) / len(sq_errors))

        self.get_logger().info('--- Final RMSE ---')
        self.get_logger().info(
            f'Position RMSE    — EKF: {rmse(self._ekf_pos_sq_errors):.4f} m   '
            f'predict-only: {rmse(self._pred_pos_sq_errors):.4f} m')
        self.get_logger().info(
            f'Orientation RMSE — EKF: {rmse(self._ekf_ori_sq_errors):.4f}°  '
            f'predict-only: {rmse(self._pred_ori_sq_errors):.4f}°')


def main(args=None):
    rclpy.init(args=args)
    node = EkfEvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._log_final_rmse()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
