#include "rl_controller_node.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>

RLControllerNode::RLControllerNode(const std::string &config_path)
    : Node("rl_controller_node") {

  if (!loadConfig(config_path)) {
    RCLCPP_ERROR(get_logger(), "Failed to load config, exiting.");
    rclcpp::shutdown();
    return;
  }

  // Subscribers
  _joint_sub = create_subscription<sensor_msgs::msg::JointState>(
      "/motion/joint_state", 10,
      std::bind(&RLControllerNode::jointStateCallback, this,
                std::placeholders::_1));

  _imu_sub = create_subscription<sensor_msgs::msg::Imu>(
      "/motion/imu", 10,
      std::bind(&RLControllerNode::imuCallback, this,
                std::placeholders::_1));

  // Publisher
  _cmd_pub = create_publisher<sensor_msgs::msg::JointState>(
      "/motion/joint_cmd", 10);

  // Control timer
  int period_ms = 1000 / _control_hz;
  _timer = create_wall_timer(
      std::chrono::milliseconds(period_ms),
      std::bind(&RLControllerNode::timerCallback, this));

  RCLCPP_INFO(get_logger(),
              "RL Controller Node started at %d Hz. DOF=%d obs=%ld action=%ld",
              _control_hz, _converter.numDof(),
              _infer->obsSize(), _infer->actionSize());
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------
bool RLControllerNode::loadConfig(const std::string &config_path) {
  try {
    YAML::Node root = YAML::LoadFile(config_path);

    _model_path = root["model_path"].as<std::string>();
    if (root["control_hz"])
      _control_hz = root["control_hz"].as<int>();

    // Load ONNX model and parse metadata
    _infer = std::make_unique<OnnxInfer>();
    if (!_infer->loadModel(_model_path, true)) {
      RCLCPP_ERROR(get_logger(), "Failed to load ONNX model: %s",
                   _model_path.c_str());
      return false;
    }

    // Build config from model metadata + YAML overrides
    const auto &meta = _infer->modelMeta();
    BeyondMimicConfig model_cfg;
    if (meta.valid()) {
      model_cfg.joint_names      = meta.joint_names;
      model_cfg.default_dof_pos  = meta.default_joint_pos;
      model_cfg.action_scales    = meta.action_scales;
      model_cfg.kp               = meta.stiffness;
      model_cfg.kd               = meta.damping;
      model_cfg.body_names       = meta.body_names;
      model_cfg.anchor_body_name = meta.anchor_body_name;
      RCLCPP_INFO(get_logger(), "Using model metadata: %d joints, %zu bodies",
                  model_cfg.numDof(), model_cfg.body_names.size());
    }

    _converter.configureFromYaml(config_path, model_cfg);

    // Read motion/timestep config
    if (root["max_timestep"])
      _max_timestep = root["max_timestep"].as<int>();
    if (root["start_timestep"])
      _start_timestep = root["start_timestep"].as<int>();

    _timestep = _start_timestep;

    RCLCPP_INFO(get_logger(),
                "Config loaded. without_state_estimator=%d use_motion_from_model=%d "
                "use_residual_action=%d max_timestep=%d",
                _converter.config().without_state_estimator,
                _converter.config().use_motion_from_model,
                _converter.config().use_residual_action,
                _max_timestep);

    return true;
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_logger(), "Config error: %s", e.what());
    return false;
  }
}

// ---------------------------------------------------------------------------
// Initialize motion data (equivalent to Python _prepare_policy)
// ---------------------------------------------------------------------------
void RLControllerNode::initializeMotion() {
  RCLCPP_INFO(get_logger(), "Initializing motion at timestep=%ld ...", _timestep);

  // Dry-run inference to get initial motion command
  auto result = _infer->dryRun(_timestep);

  if (!result.joint_pos.empty()) {
    MotionCommand cmd;
    cmd.joint_pos  = result.joint_pos;
    cmd.joint_vel  = result.joint_vel;
    cmd.body_pos_w = result.body_pos_w;
    cmd.body_quat_w = result.body_quat_w;
    _converter.setMotionCommand(cmd);

    RCLCPP_INFO(get_logger(), "Motion initialized: %zu joints, %zu bodies",
                cmd.joint_pos.size(),
                cmd.body_pos_w.size() / 3);

    // Setup initial alignment if we have body data
    if (_converter.config().numBodies() > 0 && !cmd.body_pos_w.empty()) {
      int anchor_idx = _converter.config().anchor_body_index;
      Eigen::Vector3f anchor_pos = cmd.bodyPos(anchor_idx);
      Eigen::Quaternionf anchor_quat = cmd.bodyQuat(anchor_idx);

      _converter.initAlignment().setBase(anchor_quat, anchor_pos, true, true);

      RCLCPP_INFO(get_logger(), "Initial alignment set at anchor[%d]", anchor_idx);
    }
  } else {
    RCLCPP_WARN(get_logger(), "Model did not output motion data (joint_pos empty). "
                "Is use_motion_from_model correct?");
  }

  // Also run processAction once to initialize _last_action
  _converter.processAction(result.actions);

  _initialized = true;
}

// ---------------------------------------------------------------------------
// Main control loop
// ---------------------------------------------------------------------------
void RLControllerNode::timerCallback() {
  if (!_joint_received || !_imu_received) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Waiting for joint_state + IMU data... (joint:%d imu:%d)",
                         _joint_received, _imu_received);
    return;
  }

  // Lazy initialization once we have sensor data
  if (!_initialized) {
    initializeMotion();
    return;
  }

  // --- check motion done ---
  if (_flag_motion_done) {
    // Keep publishing last command (hold position)
    return;
  }

  // --- extract current state ---
  Eigen::Quaternionf imu_quat(
      _latest_imu.orientation.w,
      _latest_imu.orientation.x,
      _latest_imu.orientation.y,
      _latest_imu.orientation.z);

  Eigen::Vector3f base_ang_vel(
      _latest_imu.angular_velocity.x,
      _latest_imu.angular_velocity.y,
      _latest_imu.angular_velocity.z);

  // --- build observation ---
  auto obs = _converter.buildObservation(
      base_ang_vel, imu_quat,
      _latest_joint_state.position,
      _latest_joint_state.velocity);

  // --- ONNX inference with timestep ---
  auto result = _infer->infer(obs, _timestep);

  // --- update motion command for next step ---
  if (!result.joint_pos.empty()) {
    MotionCommand cmd;
    cmd.joint_pos  = result.joint_pos;
    cmd.joint_vel  = result.joint_vel;
    cmd.body_pos_w = result.body_pos_w;
    cmd.body_quat_w = result.body_quat_w;
    _converter.setMotionCommand(cmd);
  }

  // --- process action: smoothing + scaling ---
  auto scaled_action = _converter.processAction(result.actions);

  // --- compute PD target ---
  auto pd_target = _converter.computePdTarget(scaled_action);

  // --- publish ---
  auto cmd = _converter.actionToJointCmd(pd_target);
  _cmd_pub->publish(cmd);

  // --- advance timestep ---
  _timestep += static_cast<int64_t>(_play_speed);

  if (_max_timestep > 0 && _timestep >= _max_timestep) {
    _play_speed = 0.0f;
    _flag_motion_done = true;
    RCLCPP_INFO(get_logger(), "Motion done at timestep %ld / %d",
                _timestep, _max_timestep);
  }
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------
void RLControllerNode::jointStateCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  _latest_joint_state = *msg;
  _joint_received = true;
}

void RLControllerNode::imuCallback(
    const sensor_msgs::msg::Imu::SharedPtr msg) {
  _latest_imu = *msg;
  _imu_received = true;
}
