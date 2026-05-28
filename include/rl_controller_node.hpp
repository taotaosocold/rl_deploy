#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <memory>

#include "onnx_infer.hpp"
#include "data_converter.hpp"

class RLControllerNode : public rclcpp::Node {
public:
  explicit RLControllerNode(const std::string &config_path);
  ~RLControllerNode() = default;

private:
  void timerCallback();
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

  bool loadConfig(const std::string &config_path);
  void initializeMotion();

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr _joint_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _imu_sub;

  // Publisher
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr _cmd_pub;

  // Timer
  rclcpp::TimerBase::SharedPtr _timer;

  // Core components
  std::unique_ptr<OnnxInfer> _infer;
  DataConverter _converter;

  // State
  sensor_msgs::msg::JointState _latest_joint_state;
  sensor_msgs::msg::Imu _latest_imu;
  bool _joint_received = false;
  bool _imu_received = false;
  bool _initialized = false;

  // Timestep tracking
  int64_t _timestep = 0;
  float _play_speed = 1.0f;
  bool _flag_motion_done = false;

  // Config
  int _control_hz = 50;
  std::string _model_path;
  int _max_timestep = -1;
  int _start_timestep = 0;
};
