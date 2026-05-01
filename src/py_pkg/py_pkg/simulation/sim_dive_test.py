import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure
from geometry_msgs.msg import PointStamped
import subprocess
import time  # For throttling logs

glider = "glider_nautilus"

# CONSTANTS

# p=p0+k*z
__std_pressure__ = 101.325  # kPa
__k__ = 9.80638 # kPa/m

class BladderMonitor(Node):
    def __init__(self):
        super().__init__('bladder_monitor')

        # Target bladder and Gazebo topic
        self.target_depth = 2.0
        self.bladder_topic = f"/model/{glider}/joint/bladder_joint/cmd_thrust"
        self.initial_bladder = -150.0

        # Threshold for surface re-entry
        self.has_reached_target_depth = False
        self.shutdown_triggered = False

        # Frequency control
        self.callback_frequency = 0.01  # secs
        self.last_callback_time = 0.0

        # Logging throttle
        self.log_frequency = 0.1  # secs
        self.last_log_time = time.time()

        # Store depth information
        self.latest_depth_msg = None
        self.last_logged_depth = None
        self.eq_counter = 0

        # Subscription to depth topic
        self.subscription = self.create_subscription(
            FluidPressure,
            f"/model/{glider}/sea_pressure",
            self.pressure_callback,
            10
        )

        # Timer to control processing frequency
        self.timer = self.create_timer(self.callback_frequency, self.process_depth)

        # Initialize bladder
        self.set_bladder(self.initial_bladder)
        self.get_logger().info(f"Initial bladder set to {self.initial_bladder}.")

    def set_bladder(self, bladder_value):
        """Send bladder command to Gazebo topic."""
        bladder_command = f"gz topic -t {self.bladder_topic} -m gz.msgs.Double -p 'data: {bladder_value}'"
        subprocess.run(bladder_command, shell=True)
        self.get_logger().info(f"Bladder set to {bladder_value}.")

    def pressure_callback(self, msg):
        """Store the latest depth message."""
        # Access the fluid pressure from the message
        pressure = msg.fluid_pressure  # This is the pressure value in Pascals
        # Convert pressure from Pascals to kPa
        # Calculate depth from pressure
        self.latest_depth_msg = (pressure - __std_pressure__) / __k__

    def process_depth(self):
        """Process the latest depth message at a controlled frequency."""
        if not self.latest_depth_msg:
            return

        current_depth = self.latest_depth_msg

        # Log only at throttled intervals
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_frequency:
            self.get_logger().info(f"Current depth: {current_depth:.2f} meters")
            self.last_log_time = current_time
            self.last_logged_depth = current_depth

        # Check bladder and node shutdown conditions
        self.check_bladder(current_depth)
        self.check_equilibrium(current_depth)

    def check_bladder(self, current_depth):
        """Check if the target depth is reached and stop the bladder."""
        if current_depth >= self.target_depth - 1e-3 and not self.has_reached_target_depth:
            self.get_logger().info(f"Target depth of {self.target_depth} meters reached. Stopping bladder...")
            self.set_bladder(0.0)
            self.has_reached_target_depth = True

    def check_equilibrium(self, current_depth):
        """Check if equilibrium is reached and shut down the node."""
        # Compare current depth with the last logged depth
        if self.last_logged_depth is not None and abs(current_depth - self.last_logged_depth) < 1e-3:
            self.eq_counter += 1
        else:
            self.eq_counter = 0

        if self.eq_counter >= 2*int(1/self.log_frequency):  # Threshold for equilibrium
            self.get_logger().info("Glider has reached equilibrium. Shutting down gracefully...")
            self.shutdown_triggered = True
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    depth_monitor = BladderMonitor()

    try:
        rclpy.spin(depth_monitor)
    except KeyboardInterrupt:
        depth_monitor.get_logger().info("Bladder monitoring interrupted by user.")
    finally:
        # Check if shutdown was already triggered
        if not depth_monitor.shutdown_triggered:
            depth_monitor.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
