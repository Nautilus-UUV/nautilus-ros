#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class StatusSubscriber : public rclcpp::Node {
public:
  StatusSubscriber() : Node("status_subscriber") {
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "status", 10,
      [this](std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", msg->data.c_str());
      });
  }

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StatusSubscriber>());
  rclcpp::shutdown();
  return 0;
}
