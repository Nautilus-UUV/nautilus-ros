import rclpy
import unittest
from std_msgs.msg import String
from py_pkg.mqtt import ROSProtocolInterface, MQTTProtocolInterface, MQTT_ROS_Bridge

class TestMQTTNode(unittest.TestCase):

    def setUp(self):
        pass

    def test_example_1(self):
        self.assertEqual(True, True)

    def test_example_2(self):
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
