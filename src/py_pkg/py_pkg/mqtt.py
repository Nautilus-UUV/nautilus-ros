"""
This node is a bridge between ROS2 and MQTT.
It subscribes to ROS2 topics and publishes to MQTT topics.
It also subscribes to MQTT topics and publishes to ROS2 topics.

Architecture:
    +-----------+     +-----------+     +-----------+     +-----------+     +-----------+
    |           |     |           |     |           |     |           |     |           |
    |MQTT Broker|<--->|MQTT Iface |<--->|MQTT/ROS   |<--->|ROS Iface  |<--->|    UUV    |
    |           |     |           |     |Bridge     |     |           |     |           |
    +-----------+     +-----------+     +-----------+     +-----------+     +-----------+


Author:
    - Thomas Bollenbach
    - Faye Dinh
Date:
    - 2025-05-07
"""

import itertools
import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from queue import PriorityQueue

import paho.mqtt.client as mqtt
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: Move to config file

MQTT_HOST = "192.168.2.1"

ros_subscr_topics = {
    # /sensor subtopics
    "/sensor/imu1": Imu,
    "/sensor/imu2": Imu,
    "/sensor/pressure_hull": Float64,
    "/sensor/pressure_tank": Float64,
    "/sensor/pressure_ext": Float64,
    "/sensor/leakage_front": Bool,
    "/sensor/leakage_back": Bool,
    # /profile subtopics
    "/profile/target_depth": Float64,
    "/profile/actual_depth": Float64,
    "/profile/actual_volume": Float64,
    # /state subtopics
    "/state/state_vec": String,
    "/state/motor": String,
    "/state/node_outage": String,
}

ros_subscr_priorities = {
    # /sensor subtopics
    "/sensor/imu1": 10,
    "/sensor/imu2": 10,
    "/sensor/pressure_hull": 6,
    "/sensor/pressure_tank": 8,
    "/sensor/pressure_ext": 10,
    "/sensor/leakage_front": 2,
    "/sensor/leakage_back": 2,
    # /profile subtopics
    "/profile/target_depth": 10,
    "/profile/actual_depth": 10,
    "/profile/actual_volume": 10,
    # /state subtopics
    "/state/state_vec": 8,
    "/state/motor": 8,
    "/state/node_outage": 2,
}

mqtt_subscr_topics = [
    "/profile/abort",
    "/profile/dive_profile",
    "/profile/start",
    "/profile/stop",
]

mqtt_publish_topics = [
    "/sensors/depth",
    "/sensors/target_depth",
    "/sensors/pose",
    "/sensors/pressure",
    "/sensors/leakage",
    "/sensors/alive",
]

mqtt_subscr_priorities = {
    "/profile/abort": 0,
    "/profile/dive_profile": 5,
    "/profile/start": 1,
    "/profile/stop": 1,
}

ros_publish_topics = {
    "/profile/dive_profile": Float64MultiArray,
    "/profile/alive": Bool,
    "/profile/start_stop": Bool,
}

ros_publish_priorities = {
    "/profile/dive_profile": 8,
    "/profile/alive": 0,
    "/profile/stop": 2,
}
mqtt_topic_translation = {
    "/profile/actual_depth": "/sensors/depth",
    "/profile/target_depth": "/sensors/target_depth",
    "/state/state_vec": "/sensors/pose",
    "/sensor/pressure_hull": "/sensors/pressure",
    "/sensor/pressure_tank": "/sensors/pressure",
    "/sensor/pressure_ext": "/sensors/pressure",
    "/sensor/leakage_front": "/sensors/leakage",
    "/sensor/leakage_back": "/sensors/leakage",
    "/sensor/imu1": "/sensors/imu",
    "/sensor/imu2": "/sensors/imu",
}


class ProtocolInterface(ABC):
    """
    This is an abstract class for the MQTT and ROS2 protocols.
    It defines the interface for the protocols.
    """

    @abstractmethod
    def send():
        pass

    @abstractmethod
    def receive():
        pass


class MQTTProtocolInterface(ProtocolInterface):
    """
    This class creates a MQTT client and connects to the MQTT broker.
    It also subscribes to the topics and receives messages from the MQTT broker.
    It also sends messages to the MQTT broker.
    It uses a priority queue to store the messages.
    """

    def __init__(self):
        logger.info("[MQTT] Initializing MQTTProtocolInterface…")
        self.inbound_queue = PriorityQueue(maxsize=100)
        self.counter = itertools.count()
        self.running = False

        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="ros2_mqtt_bridge",
            clean_session=False,
            protocol=mqtt.MQTTv311,
            transport="tcp",
        )

        # Set callback functions
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_subscribe = self.on_subscribe
        self.mqtt_client.on_message = lambda client, userdata, msg: (
            self.receive(msg.topic, msg.payload)
        )

    def connect(self):
        """
        This function connects to the MQTT broker.
        """
        logger.info("[MQTT] Connecting to MQTT broker…")
        result = self.mqtt_client.connect(
            host=MQTT_HOST,
            port=1883,
            keepalive=60,
        )
        if result == mqtt.MQTT_ERR_SUCCESS:
            logger.info("[MQTT] Socket connection successful, awaiting CONNACK…")
        else:
            logger.error(
                f"[MQTT] Socket connection failed: {mqtt.error_string(result)}"
            )

        logger.info("[MQTT] Starting network loop…")
        self.mqtt_client.loop_start()
        self.running = True
        logger.info("[MQTT] Network loop started")

    def stop(self):
        """
        This function stops the MQTT client.
        """
        logger.info("[MQTT] Stopping network loop…")
        self.mqtt_client.loop_stop()
        logger.info("[MQTT] Network loop stopped")
        logger.info("[MQTT] Disconnecting from broker…")
        self.mqtt_client.disconnect()
        logger.info("[MQTT] Disconnected from broker")
        self.running = False

    def on_connect(self, client, userdata, connect_flags, reason_code, properties):
        """
        This function is called when the MQTT client connects to the broker.
        """
        if not reason_code:
            logger.error(
                f"[MQTT] Connection to the MQTT broker failed: {reason_code.getName()}"
            )
        else:
            logger.info("[MQTT] Connection to the MQTT broker successful")
            logger.info("[MQTT] Subscribing to topics…")

            # Subscribing to topics
            self.mqtt_client.subscribe([(topic, 1) for topic in mqtt_subscr_topics])

            logger.info("[MQTT] Subscribed to topics")

    def on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ):
        """
        This function is called when the MQTT client disconnects from the broker.
        """
        logger.info(
            f"[MQTT] Disconnected from the MQTT broker: {mqtt.error_string(reason_code)}"
        )
        self.inbound_queue.put_nowait(
            (0, next(self.counter), ("/profile/alive", False))
        )

    def on_subscribe(self, client, userdata, mid, reason_code_list, properties):
        """
        This function is called when the MQTT client subscribes to a topic.
        """
        logger.info(f"[MQTT] Subscribed: {mid} {reason_code_list}")

    def send(self, topic, msg):
        """
        This function sends a message to the MQTT broker.
        Args:
            - topic (Str): The topic to send the message to.
            - msg (JSON): The message to send.
        """
        if topic not in mqtt_publish_topics:
            logger.warning(
                f"[MQTT] Sending message to unknown topic {topic}, dropping message"
            )
            return
        self.mqtt_client.publish(topic, msg)

    def receive(self, topic, msg):
        """
        This is a callback function for the MQTT client.
        It receives a message from the MQTT broker and puts it in the priority queue.
        Args:
            - topic (Str): The topic the message was received from.
            - msg (JSON): The message received.
        """
        if topic not in mqtt_subscr_topics:
            logger.warning(
                f"[MQTT] Received message for unknown topic {topic}, dropping message"
            )
            return
        if self.inbound_queue.full():
            logger.warning("[MQTT] Outbound queue is full, dropping message")
        else:
            try:
                msg_str = msg.decode("utf-8")
                self.inbound_queue.put_nowait(
                    (
                        mqtt_subscr_priorities[topic],  # priority
                        next(self.counter),  # tie breaker
                        (topic, msg_str),
                    )  # message
                )
                self.mqtt_client.publish(
                    topic="/commands/ack", payload=self.get_msg_id(msg_str)
                )
            except UnicodeDecodeError:
                logger.warning(
                    f"[MQTT] Failed to decode message for topic {topic}, dropping message"
                )
                return

    def get_msg_id(self, msg_str):
        """
        This function extracts the message ID from the message.
        """
        msg_json = json.loads(msg_str)
        response_msg = {"command_id": msg_json["command_id"]}
        return json.dumps(response_msg)


class ROSProtocolInterface(Node, ProtocolInterface):
    """
    This class creates a ROS2 node and connects to the ROS2 broker.
    It also subscribes to the topics and receives messages from the ROS2 broker.
    It also sends messages to the ROS2 broker.
    It uses a priority queue to store the messages.
    """

    def __init__(self):
        logger.info("[ROS] Initializing ROSProtocolInterface…")
        super().__init__("ros2_mqtt_bridge")
        self.inbound_queue = PriorityQueue(maxsize=100)
        self.counter = itertools.count()
        # Create publishers for each topic in ros_publish_topics
        logger.info("[ROS] Creating publishers…")
        self.ros_publishers = {}
        for topic, msg_type in ros_publish_topics.items():
            self.ros_publishers[topic] = self.create_publisher(msg_type, topic, 10)

        # Create subscribers for each topic in ros_subscr_topics
        logger.info("[ROS] Creating subscribers…")
        self.ros_subscribers = {}
        for topic, msg_type in ros_subscr_topics.items():
            self.ros_subscribers[topic] = self.create_subscription(
                msg_type,
                topic,
                lambda msg, t=topic: self.receive(t, msg),
                10,
            )
        logger.info("[ROS] Subscribers created")
        # Create client calling start service to mcn

    #     self.cli = self.create_client(SetBool, "start_and_abort")
    #     while not self.cli.wait_for_service(timeout_sec=1.0):
    #         self.get_logger().info('start and stop service not available, waiting again...')
    #     self.req = SetBool.Request()

    # def send_start_stop_call(self, value):
    #     self.req.data = value
    #     return self.cli.call_async(self.req)

    def stop(self):
        """
        This function stops the ROS2 node.
        """
        logger.info("[ROS] Stopping ROSProtocolInterface…")
        self.destroy_node()
        logger.info("[ROS] ROSProtocolInterface stopped")

    def send(self, topic, data):
        """
        This function sends a message to the ROS2 broker.
        Args:
            - topic (Str): The topic to send the message to.
            - data (ROS2 message): The message to send.
        """
        if topic not in ros_publish_topics:
            logger.warning(
                f"[ROS] Sending message to unknown topic {topic}, dropping message"
            )
            return
        msg = ros_publish_topics[topic](data=data)
        self.ros_publishers[topic].publish(msg)

    def receive(self, topic, msg):
        """
        This is a callback function for the ROS2 node.
        It receives a message from the ROS2 broker and puts it in the priority queue.
        Args:
            - topic (Str): The topic the message was received from.
            - msg (ROS2 message): The message received.
        """
        if topic not in ros_subscr_topics:
            logger.warning(
                f"[ROS] Received message for unknown topic {topic}, dropping message"
            )
            return
        if self.inbound_queue.full():
            logger.warning("[ROS] Outbound queue is full, dropping message")
        else:
            try:
                self.inbound_queue.put_nowait(
                    (
                        ros_subscr_priorities[topic],  # priority
                        next(self.counter),  # tie breaker
                        (topic, msg),
                    )  # message
                )
            except Exception as e:
                logger.error(f"[ROS] Error putting message in queue: {e}")


class MQTT_ROS_Bridge:
    """
    This class creates a bridge between MQTT and ROS2. It instantiates the MQTT and ROS2 nodes.
    It also implements the translation between MQTT and ROS2 messages.
    It also runs the ROS2 executor, the MQTT to ROS2 translation, and the ROS2 to MQTT
    translation in separate threads.
    """

    def __init__(self):
        self.mqtt_protocol = MQTTProtocolInterface()
        self.ros_protocol = ROSProtocolInterface()
        self.ros_executor = SingleThreadedExecutor()
        self.ros_executor.add_node(self.ros_protocol)

    def run(self):
        """
        This function runs the bridge. It connects to the MQTT broker, starts the ROS2 executor,
        and starts the threads for the MQTT to ROS2 and ROS2 to MQTT translations.
        The ROS2 executor is run in a separate thread to avoid blocking the main thread.
        """
        self.mqtt_protocol.connect()
        # Run the ROS2 executor in a separate thread to avoid blocking the main thread
        spin_thread = threading.Thread(
            target=spin_executor,
            args=(self.ros_executor,),
            daemon=True,
        )
        spin_thread.start()
        try:
            # Run the MQTT to ROS2 translation in a separate thread to avoid blocking
            # the main thread while waiting for messages
            mqtt2ros_thread = threading.Thread(
                target=self.mqtt_to_ros,
                daemon=True,
            )
            # Run the ROS2 to MQTT translation in a separate thread to avoid blocking
            # the main thread while waiting for messages
            ros2mqtt_thread = threading.Thread(
                target=self.ros_to_mqtt,
                daemon=True,
            )
            ros2mqtt_thread.start()
            mqtt2ros_thread.start()
            while rclpy.ok() and self.mqtt_protocol.running:
                pass
        except KeyboardInterrupt:
            logger.info("[MQTT_ROS_Bridge] Keyboard interrupt, stopping…")
        finally:
            self.stop()
            mqtt2ros_thread.join()
            ros2mqtt_thread.join()
            spin_thread.join()

    def mqtt_to_ros(self):
        """
        This function extracts messages from the MQTT priority queue and translates them
        to ROS2 messages.
        It then sends the ROS2 messages to the ROS2 broker.
        Run in a separate thread.
        """
        queue = self.mqtt_protocol.inbound_queue
        while self.mqtt_protocol.running:
            _, _, (topic, msg) = queue.get()
            ros_msg = self.json_to_msg(topic, msg)
            if ros_msg:
                self.ros_protocol.send(ros_msg[0], ros_msg[1])

    def json_to_msg(self, topic, json_data):
        """
        This function translates MQTT messages to ROS2 messages.

        Args:
            - topic (Str): The topic the message was received from.
            - json_data (JSON): The message received.

        Returns:
            - topic (Str): The topic to send the message to.
            - data (ROS2 message): The message to send.
        """
        if topic == "/profile/abort":
            logger.info("[MQTT_ROS_Bridge] Received abort message")
            return ("/profile/alive", False)
        if topic == "/profile/dive_profile":
            json_msg = json.loads(json_data)
            waypoints = []
            for waypoint in json_msg["data"]["waypoints"]:
                waypoints.append(waypoint["depth"])
                waypoints.append(waypoint["pause_duration"])
            return ("/profile/dive_profile", waypoints)
        if topic == "/profile/start":
            # future = self.ros_protocol.send_start_stop_call(True)
            # rclpy.spin_until_future_complete(self,future)
            # response = future.result()
            # self.get_logger().info(f"sending start command success:{response}")
            return ("/profile/alive", True)
            pass
        if topic == "/profile/stop":
            # future = self.ros_protocol.send_start_stop_call(False)
            # rclpy.spin_until_future_complete(self,future)
            # response = future.result()
            # self.get_logger().info(f"sending stop command success:{response}")
            return ("/profile/stop", True)
            pass
        return None

    def ros_to_mqtt(self):
        """
        This function extracts messages from the ROS2 priority queue and translates them
        to MQTT messages.
        It then sends the MQTT messages to the MQTT broker.
        Run in a separate thread.
        """
        queue = self.ros_protocol.inbound_queue
        while rclpy.ok():
            _, _, (topic, msg) = queue.get()
            mqtt_msg = self.msg_to_json(topic, msg)
            if mqtt_msg:
                self.mqtt_protocol.send(mqtt_msg[0], mqtt_msg[1])

    def msg_to_json(self, topic, msg):
        """
        This function translates ROS2 messages to MQTT messages.
        Args:
            - topic (Str): The topic the message was received from.
            - msg (ROS2 message): The message received.

        Returns:
            - topic (Str): The topic to send the message to.
            - data (JSON): The message to send.
        """
        pressure_locations = [
            "/sensor/pressure_hull",
            "/sensor/pressure_tank",
            "/sensor/pressure_ext",
        ]
        leakage_locations = ["/sensor/leakage_front", "/sensor/leakage_back"]
        imu_locations = ["/sensor/imu1", "/sensor/imu2"]

        timestamp = datetime.now().isoformat()
        dict_msg = {"record_datetime": timestamp}
        if topic == "/profile/actual_depth":
            dict_msg["depth"] = msg.data
        elif topic == "/profile/target_depth":
            dict_msg["depth"] = msg.data
        elif topic == "/state/state_vec":
            state_vec_idxs = {
                "x": 0,
                "y": 2,
                "z": 4,
                "qw": 6,
                "qx": 7,
                "qy": 8,
                "qz": 9,
            }
            for key in state_vec_idxs:
                dict_msg[key] = msg.x[state_vec_idxs[key]]
        elif topic in pressure_locations:
            dict_msg["pressure"] = msg.data
            _, _, dict_msg["location"] = topic.partition("_")
        elif topic in leakage_locations:
            dict_msg["has_leak"] = msg.data
            _, _, dict_msg["location"] = topic.partition("_")
        elif topic in imu_locations:
            dict_msg["imu_num"] = int(topic[-1])
            dict_msg["angular_velocity"] = msg.angular_velocity
            dict_msg["linear_acceleration"] = msg.linear_acceleration
        if dict_msg:
            return (mqtt_topic_translation[topic], json.dumps(dict_msg))
        return None

    def stop(self):
        """
        This function stops the bridge. It stops the MQTT client,
        the ROS2 node, and the ROS2 executor.
        """
        logger.info("[MQTT_ROS_Bridge] Stopping…")
        self.mqtt_protocol.stop()
        self.ros_protocol.stop()
        rclpy.shutdown()
        logger.info("[MQTT_ROS_Bridge] Stopped")


def spin_executor(executor):
    """
    This function spins the ROS2 executor.
    """
    executor.spin()


def main(args=None):
    """
    This function initializes the ROS2 node, creates the bridge, and runs it.
    It is the entry point of the node.
    """
    rclpy.init(args=args)
    bridge = MQTT_ROS_Bridge()
    bridge.run()


if __name__ == "__main__":
    main()
