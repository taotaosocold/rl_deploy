#include "rl_controller_node.hpp"
#include <csignal>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  std::string config_path = "config/rl_config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  auto node = std::make_shared<RLControllerNode>(config_path);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
