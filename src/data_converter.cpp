#include "data_converter.hpp"
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <algorithm>
#include <iostream>

// ============================================================================
// DoFAdapter
// ============================================================================
void DoFAdapter::build(const std::vector<std::string> &src_names,
                        const std::vector<std::string> &tar_names,
                        const std::map<std::string, std::string> &name_map) {
  _src_len = static_cast<int>(src_names.size());
  _tar_len = static_cast<int>(tar_names.size());

  _src_indices.clear();
  _tar_indices.clear();

  // Build a reverse index for tar names for fast lookup
  std::map<std::string, int> tar_index;
  for (int j = 0; j < _tar_len; j++) {
    tar_index[tar_names[j]] = j;
  }

  for (int si = 0; si < _src_len; si++) {
    const std::string &src_name = src_names[si];

    // 1) Try explicit name map first
    auto map_it = name_map.find(src_name);
    if (map_it != name_map.end()) {
      auto ti = tar_index.find(map_it->second);
      if (ti != tar_index.end()) {
        _src_indices.push_back(si);
        _tar_indices.push_back(ti->second);
        continue;
      }
    }

    // 2) Fallback: same-name matching
    auto ti = tar_index.find(src_name);
    if (ti != tar_index.end()) {
      _src_indices.push_back(si);
      _tar_indices.push_back(ti->second);
    }
  }

  std::cout << "[DoFAdapter] Mapped " << _src_indices.size()
            << " / " << _src_len << " → " << _tar_len
            << " joints (name_map has " << name_map.size() << " entries)"
            << std::endl;
}

std::vector<float> DoFAdapter::fit(const std::vector<float> &data,
                                    const std::vector<float> &template_vals) const {
  std::vector<float> result;

  if (!template_vals.empty()) {
    result = template_vals;
  } else {
    result.assign(_tar_len, 0.0f);
  }

  int n = static_cast<int>(_src_indices.size());
  for (int k = 0; k < n; k++) {
    int si = _src_indices[k];
    int ti = _tar_indices[k];
    if (si < static_cast<int>(data.size())) {
      result[ti] = data[si];
    }
  }

  return result;
}

// ============================================================================
// TransformAlignment
// ============================================================================
void TransformAlignment::setBase(const Eigen::Quaternionf &quat,
                                  const Eigen::Vector3f &pos,
                                  bool yaw_only, bool xy_only) {
  yaw_only_ = yaw_only;
  xy_only_ = xy_only;

  if (yaw_only_) {
    Eigen::Vector3f euler = quat.toRotationMatrix().eulerAngles(0, 1, 2);
    euler.x() = 0.0f;
    euler.y() = 0.0f;
    R_base_ = Eigen::AngleAxisf(euler.z(), Eigen::Vector3f::UnitZ());
  } else {
    R_base_ = quat;
  }

  if (xy_only_) {
    p_base_ = Eigen::Vector3f(pos.x(), pos.y(), 0.0f);
  } else {
    p_base_ = pos;
  }
}

Eigen::Quaternionf TransformAlignment::alignQuat(const Eigen::Quaternionf &q) const {
  return R_base_.inverse() * q;
}

Eigen::Vector3f TransformAlignment::alignPos(const Eigen::Vector3f &p) const {
  return R_base_.inverse() * (p - p_base_);
}

void TransformAlignment::alignTransform(Eigen::Quaternionf &q,
                                         Eigen::Vector3f &p) const {
  q = alignQuat(q);
  p = alignPos(p);
}

// ============================================================================
// Math utilities
// ============================================================================
Eigen::Matrix3f matrixFromQuat(const Eigen::Quaternionf &q) {
  return q.toRotationMatrix();
}

void subtractFrameTransforms(
    const Eigen::Vector3f &t01, const Eigen::Quaternionf &q01,
    const Eigen::Vector3f &t02, const Eigen::Quaternionf &q02,
    Eigen::Vector3f &t12, Eigen::Quaternionf &q12) {

  Eigen::Quaternionf q10 = q01.inverse();
  Eigen::Vector3f t_rel = t02 - t01;
  t12 = q10 * t_rel;
  q12 = q10 * q02;
}

// ============================================================================
// DataConverter
// ============================================================================
void DataConverter::configure(const BeyondMimicConfig &cfg) {
  _cfg = cfg;
  _cfg.setAnchorBodyIndex();
  _last_action.assign(cfg.numDof(), 0.0f);

  // Build DoFAdapters
  const auto &robot_names = _cfg.robot_joint_names.empty()
                                ? _cfg.joint_names
                                : _cfg.robot_joint_names;

  if (_cfg.robot_joint_names.empty()) {
    std::cout << "[DataConverter] No robot_joint_names set, assuming 1:1 "
              << "mapping to model joint order (" << _cfg.numDof() << " DOF)"
              << std::endl;
  } else {
    _robot2model.build(robot_names, _cfg.joint_names, _cfg.joint_name_map);

    // Build reverse mapping for model→robot direction
    std::map<std::string, std::string> reverse_map;
    for (const auto &[ros2_name, model_name] : _cfg.joint_name_map) {
      reverse_map[model_name] = ros2_name;
    }
    _model2robot.build(_cfg.joint_names, robot_names, reverse_map);
  }
}

void DataConverter::configureFromYaml(const std::string &yaml_path,
                                       const BeyondMimicConfig &model_meta) {
  YAML::Node root = YAML::LoadFile(yaml_path);

  BeyondMimicConfig cfg;

  // --- start from model metadata ---
  if (model_meta.numDof() > 0) {
    cfg.joint_names      = model_meta.joint_names;
    cfg.default_dof_pos  = model_meta.default_dof_pos;
    cfg.kp               = model_meta.kp;
    cfg.kd               = model_meta.kd;
    cfg.action_scales    = model_meta.action_scales;
    cfg.body_names       = model_meta.body_names;
    cfg.anchor_body_name = model_meta.anchor_body_name;
  }

  // --- robot joint order override ---
  if (root["robot_joint_names"]) {
    cfg.robot_joint_names = root["robot_joint_names"].as<std::vector<std::string>>();
  }

  // --- explicit joint name mapping: ros2_name → model_name ---
  if (root["joint_name_map"]) {
    for (const auto &kv : root["joint_name_map"]) {
      cfg.joint_name_map[kv.first.as<std::string>()] = kv.second.as<std::string>();
    }
  }

  // --- yaml overrides for model metadata ---
  if (root["joint_names"]) {
    cfg.joint_names = root["joint_names"].as<std::vector<std::string>>();
  }
  if (root["default_dof_pos"]) {
    cfg.default_dof_pos = root["default_dof_pos"].as<std::vector<float>>();
  }
  if (root["kp"]) {
    cfg.kp = root["kp"].as<std::vector<float>>();
  }
  if (root["kd"]) {
    cfg.kd = root["kd"].as<std::vector<float>>();
  }
  if (root["action_scales"] && root["action_scales"].size() > 0) {
    cfg.action_scales = root["action_scales"].as<std::vector<float>>();
  }
  if (root["body_names"]) {
    cfg.body_names = root["body_names"].as<std::vector<std::string>>();
  }
  if (root["anchor_body_name"]) {
    cfg.anchor_body_name = root["anchor_body_name"].as<std::string>();
  }

  // --- flags ---
  if (root["without_state_estimator"])
    cfg.without_state_estimator = root["without_state_estimator"].as<bool>();
  if (root["use_motion_from_model"])
    cfg.use_motion_from_model = root["use_motion_from_model"].as<bool>();
  if (root["use_residual_action"])
    cfg.use_residual_action = root["use_residual_action"].as<bool>();
  if (root["override_robot_anchor_pos"])
    cfg.override_robot_anchor_pos = root["override_robot_anchor_pos"].as<bool>();

  // --- action ---
  if (root["action_beta"])
    cfg.action_beta = root["action_beta"].as<float>();
  if (root["clip_actions"])
    cfg.clip_actions = root["clip_actions"].as<float>();

  // --- motion ---
  if (root["start_timestep"])
    cfg.start_timestep = root["start_timestep"].as<int>();
  if (root["max_timestep"])
    cfg.max_timestep = root["max_timestep"].as<int>();

  // --- fallbacks ---
  int N = cfg.numDof();
  if (cfg.action_scales.empty())    cfg.action_scales.assign(N, 0.25f);
  if (cfg.default_dof_pos.empty())  cfg.default_dof_pos.assign(N, 0.0f);
  if (cfg.kp.empty())               cfg.kp.assign(N, 100.0f);
  if (cfg.kd.empty())               cfg.kd.assign(N, 10.0f);

  configure(cfg);

  std::cout << "[DataConverter] Configured: " << N << " DOF, "
            << cfg.numBodies() << " bodies, anchor=" << cfg.anchor_body_name
            << " (idx=" << cfg.anchor_body_index << ")"
            << ", without_state_estimator=" << cfg.without_state_estimator
            << ", use_motion_from_model=" << cfg.use_motion_from_model
            << ", use_residual_action=" << cfg.use_residual_action
            << ", action_beta=" << cfg.action_beta
            << std::endl;
}

// ---------------------------------------------------------------------------
// Build observation (BeyondMimic format)
// Input dof_pos/dof_vel are in ROBOT order → remapped to MODEL order via
// _robot2model.
// ---------------------------------------------------------------------------
std::vector<float> DataConverter::buildObservation(
    const Eigen::Vector3f &base_ang_vel,
    const Eigen::Quaternionf &imu_quat,
    const std::vector<double> &dof_pos,
    const std::vector<double> &dof_vel) {

  int N = _cfg.numDof();

  // Remap dof_pos / dof_vel from robot order to model policy order
  std::vector<float> dof_pos_f(dof_pos.begin(), dof_pos.end());
  std::vector<float> dof_vel_f(dof_vel.begin(), dof_vel.end());

  std::vector<float> dof_pos_model = _robot2model.fit(dof_pos_f);
  std::vector<float> dof_vel_model = _robot2model.fit(dof_vel_f);

  // Also remap default_dof_pos from model order to match
  // (default_dof_pos is already in model order, so no remap needed)

  std::vector<float> obs;

  // --- 1. Command (reference motion joint_pos + joint_vel, MODEL order) ---
  if (_command.empty()) {
    obs.insert(obs.end(), N * 2, 0.0f);
  } else {
    for (int i = 0; i < N; i++)
      obs.push_back(_command.joint_pos[i]);
    for (int i = 0; i < N; i++)
      obs.push_back(_command.joint_vel[i]);
  }

  // --- 2. Motion anchor orientation (6 values) ---
  if (!_command.empty() && _cfg.numBodies() > 0 && _cfg.anchor_body_index >= 0) {
    Eigen::Vector3f anchor_pos_w = _command.bodyPos(_cfg.anchor_body_index);
    Eigen::Quaternionf anchor_quat_w = _command.bodyQuat(_cfg.anchor_body_index);

    _init_align.alignTransform(anchor_quat_w, anchor_pos_w);

    Eigen::Quaternionf robot_anchor_quat = imu_quat;
    Eigen::Vector3f robot_anchor_pos = Eigen::Vector3f::Zero();

    if (_cfg.override_robot_anchor_pos) {
      robot_anchor_pos = anchor_pos_w;
    }

    Eigen::Vector3f rel_pos;
    Eigen::Quaternionf rel_quat;
    subtractFrameTransforms(robot_anchor_pos, robot_anchor_quat,
                            anchor_pos_w, anchor_quat_w,
                            rel_pos, rel_quat);

    if (!_cfg.without_state_estimator) {
      obs.push_back(rel_pos.x());
      obs.push_back(rel_pos.y());
      obs.push_back(rel_pos.z());
    }

    Eigen::Matrix3f mat = matrixFromQuat(rel_quat);
    // numpy mat[:, :2].flatten() in C order = [m00,m01,m10,m11,m20,m21]
    for (int r = 0; r < 3; r++) {
      obs.push_back(mat(r, 0));
      obs.push_back(mat(r, 1));
    }
  } else {
    if (!_cfg.without_state_estimator) {
      obs.insert(obs.end(), 3, 0.0f);
    }
    obs.insert(obs.end(), 6, 0.0f);
  }

  // --- 3. Base linear velocity: excluded (without_state_estimator=True) ---

  // --- 4. Base angular velocity (3) ---
  obs.push_back(base_ang_vel.x());
  obs.push_back(base_ang_vel.y());
  obs.push_back(base_ang_vel.z());

  // --- 5. Joint position relative to default (N, MODEL order, relative to model default) ---
  for (int i = 0; i < N; i++) {
    float v = dof_pos_model[i] - _cfg.default_dof_pos[i];
    obs.push_back(v);
  }

  // --- 6. Joint velocity (N, MODEL order) ---
  for (int i = 0; i < N; i++) {
    obs.push_back(dof_vel_model[i]);
  }

  // --- 7. Last action (N, MODEL order) ---
  for (int i = 0; i < N; i++) {
    obs.push_back(_last_action[i]);
  }

  return obs;
}

// ---------------------------------------------------------------------------
// Process action: smoothing + per-joint scaling (MODEL order)
// ---------------------------------------------------------------------------
std::vector<float> DataConverter::processAction(const std::vector<float> &raw_action) {
  int N = _cfg.numDof();
  std::vector<float> smoothed(N);

  for (int i = 0; i < N; i++) {
    smoothed[i] = (1.0f - _cfg.action_beta) * _last_action[i]
                + _cfg.action_beta * raw_action[i];
  }

  std::vector<float> scaled(N);
  float clip = _cfg.clip_actions;
  for (int i = 0; i < N; i++) {
    float a = std::clamp(smoothed[i], -clip, clip);
    scaled[i] = a * _cfg.action_scales[i];
  }

  _last_action = smoothed;
  return scaled;
}

// ---------------------------------------------------------------------------
// PD target: MODEL order → remap to ROBOT order
// ---------------------------------------------------------------------------
std::vector<float> DataConverter::computePdTarget(const std::vector<float> &scaled_action) {
  int N = _cfg.numDof();

  // Compute in MODEL order
  std::vector<float> target_model(N);

  if (_cfg.use_residual_action && !_command.empty()) {
    for (int i = 0; i < N; i++) {
      target_model[i] = scaled_action[i] + _command.joint_pos[i];
    }
  } else {
    for (int i = 0; i < N; i++) {
      target_model[i] = _cfg.default_dof_pos[i] + scaled_action[i];
    }
  }

  // Remap MODEL → ROBOT order for publishing
  // Use robot_joint_names's default_dof_pos as template if available
  const auto &robot_names = _cfg.robot_joint_names.empty()
                                ? _cfg.joint_names
                                : _cfg.robot_joint_names;

  if (_cfg.robot_joint_names.empty()) {
    // 1:1 mapping — no remap needed
    return target_model;
  } else {
    // Build template from model default_pos remapped to robot order
    // Robot default_pos: _model2robot.fit(_, robot_dof_pos)
    // Actually, use model2robot with empty template (zeros for unmatched joints)
    return _model2robot.fit(target_model);
  }
}

// ---------------------------------------------------------------------------
// Convert PD target (ROBOT order) → JointState message
// ---------------------------------------------------------------------------
sensor_msgs::msg::JointState DataConverter::actionToJointCmd(
    const std::vector<float> &pd_target) {

  sensor_msgs::msg::JointState cmd;
  cmd.header.stamp = rclcpp::Clock().now();

  // Use robot_joint_names for the published message header
  const auto &pub_names = _cfg.robot_joint_names.empty()
                              ? _cfg.joint_names
                              : _cfg.robot_joint_names;

  int N = std::min((int)pd_target.size(), (int)pub_names.size());

  for (int i = 0; i < N; i++) {
    cmd.name.push_back(pub_names[i]);
    cmd.position.push_back(pd_target[i]);
    cmd.velocity.push_back(0.0);
    cmd.effort.push_back(0.0);
  }

  return cmd;
}
