#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <string>
#include <map>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>

// ---------------------------------------------------------------------------
// Motion data snapshot — stored from model output, fed as "command" next step
// ---------------------------------------------------------------------------
struct MotionCommand {
  std::vector<float> joint_pos;
  std::vector<float> joint_vel;
  std::vector<float> body_pos_w;   // [num_bodies * 3]
  std::vector<float> body_quat_w;  // [num_bodies * 4]  w,x,y,z

  // accessors by body index
  Eigen::Vector3f bodyPos(int idx) const {
    return Eigen::Vector3f(body_pos_w[idx*3], body_pos_w[idx*3+1], body_pos_w[idx*3+2]);
  }
  Eigen::Quaternionf bodyQuat(int idx) const {
    // stored as w,x,y,z → Eigen constructor is w,x,y,z
    return Eigen::Quaternionf(body_quat_w[idx*4], body_quat_w[idx*4+1],
                              body_quat_w[idx*4+2], body_quat_w[idx*4+3]);
  }
  bool empty() const { return joint_pos.empty(); }
};

// ---------------------------------------------------------------------------
// DoFAdapter — name-based joint reordering (ported from RoboJuDo DoFAdapter)
// ---------------------------------------------------------------------------
class DoFAdapter {
public:
  DoFAdapter() = default;

  /// Build index maps between src and tar joint name lists.
  /// If name_map is provided, uses it to translate src names → tar names.
  /// Otherwise falls back to same-name matching.
  void build(const std::vector<std::string> &src_names,
             const std::vector<std::string> &tar_names,
             const std::map<std::string, std::string> &name_map = {});

  /// Reorder data from src order to tar order.
  /// If template is non-empty, unmapped joints keep template values.
  std::vector<float> fit(const std::vector<float> &data,
                         const std::vector<float> &template_vals = {}) const;

  int srcLen() const { return _src_len; }
  int tarLen() const { return _tar_len; }

private:
  std::vector<int> _src_indices;
  std::vector<int> _tar_indices;
  int _src_len = 0;
  int _tar_len = 0;
};

// ---------------------------------------------------------------------------
// Config for BeyondMimic data pipeline
// ---------------------------------------------------------------------------
struct BeyondMimicConfig {
  // --- model joint order (from ONNX metadata) ---
  std::vector<std::string> joint_names;       // model policy joint order
  std::vector<float> default_dof_pos;         // in model order
  std::vector<float> kp;                      // in model order
  std::vector<float> kd;                      // in model order
  std::vector<float> action_scales;           // in model order

  // --- robot joint order (from ROS2 /motion/joint_state) ---
  // If empty, assumed same as model joint_names order (1:1 index mapping).
  // If set, DoFAdapter remaps robot↔model using joint_name_map.
  std::vector<std::string> robot_joint_names;

  // --- explicit name mapping: ros2_name → model_name ---
  // E.g. {"leg_l1_joint": "left_leg_pelvic_pitch_joint"}
  // If a name is in robot_joint_names but NOT in this map, same-name
  // matching is attempted as fallback.
  std::map<std::string, std::string> joint_name_map;

  // --- motion handling ---
  std::string anchor_body_name;
  std::vector<std::string> body_names;
  int anchor_body_index = 0;

  // --- flags (match RoboJuDo BeyondMimicPolicyCfg) ---
  bool without_state_estimator = true;
  bool use_motion_from_model = true;
  bool use_residual_action = false;
  bool override_robot_anchor_pos = true;

  // --- action post-processing ---
  float action_beta = 1.0f;
  float clip_actions = 100.0f;

  int max_timestep = -1;
  int start_timestep = 0;

  int numDof() const { return static_cast<int>(joint_names.size()); }
  int numBodies() const { return static_cast<int>(body_names.size()); }
  int numRobotDof() const {
    return robot_joint_names.empty() ? numDof()
                                     : static_cast<int>(robot_joint_names.size());
  }

  void setAnchorBodyIndex() {
    for (size_t i = 0; i < body_names.size(); i++) {
      if (body_names[i] == anchor_body_name) {
        anchor_body_index = static_cast<int>(i);
        return;
      }
    }
    anchor_body_index = 0;
  }
};

// ---------------------------------------------------------------------------
// TransformAlignment — yaw-only + xy-only alignment (ported from Python)
// ---------------------------------------------------------------------------
class TransformAlignment {
public:
  TransformAlignment() = default;

  void setBase(const Eigen::Quaternionf &quat, const Eigen::Vector3f &pos,
               bool yaw_only = true, bool xy_only = true);

  Eigen::Quaternionf alignQuat(const Eigen::Quaternionf &q) const;
  Eigen::Vector3f alignPos(const Eigen::Vector3f &p) const;
  void alignTransform(Eigen::Quaternionf &q, Eigen::Vector3f &p) const;

private:
  Eigen::Quaternionf R_base_ = Eigen::Quaternionf::Identity();
  Eigen::Vector3f p_base_ = Eigen::Vector3f::Zero();
  bool yaw_only_ = true;
  bool xy_only_ = true;
};

// ---------------------------------------------------------------------------
// Math utilities (ported from RoboJuDo util_func.py)
// ---------------------------------------------------------------------------
Eigen::Matrix3f matrixFromQuat(const Eigen::Quaternionf &q);

void subtractFrameTransforms(
    const Eigen::Vector3f &t01, const Eigen::Quaternionf &q01,
    const Eigen::Vector3f &t02, const Eigen::Quaternionf &q02,
    Eigen::Vector3f &t12, Eigen::Quaternionf &q12);

// ---------------------------------------------------------------------------
// DataConverter — builds BeyondMimic observations, processes actions
// ---------------------------------------------------------------------------
class DataConverter {
public:
  DataConverter() = default;

  void configure(const BeyondMimicConfig &cfg);
  void configureFromYaml(const std::string &yaml_path,
                         const BeyondMimicConfig &model_meta = {});

  /// Build observation matching RoboJuDo BeyondMimic format.
  /// dof_pos / dof_vel are in ROBOT joint order; they are remapped to model
  /// order internally via DoFAdapter.
  std::vector<float> buildObservation(
      const Eigen::Vector3f &base_ang_vel,
      const Eigen::Quaternionf &imu_quat,
      const std::vector<double> &dof_pos,
      const std::vector<double> &dof_vel);

  /// Process raw model action (model order) through smoothing + per-joint scaling.
  /// Returns scaled action in MODEL order.
  std::vector<float> processAction(const std::vector<float> &raw_action);

  /// Compute PD target in ROBOT joint order.
  std::vector<float> computePdTarget(const std::vector<float> &scaled_action);

  /// Convert pd_target (robot order) to JointState command message.
  sensor_msgs::msg::JointState actionToJointCmd(
      const std::vector<float> &pd_target);

  /// Store motion data from model output for next step's observation
  void setMotionCommand(const MotionCommand &cmd) { _command = cmd; }
  const MotionCommand &motionCommand() const { return _command; }

  /// Access config
  const BeyondMimicConfig &config() const { return _cfg; }
  BeyondMimicConfig &config() { return _cfg; }

  /// Initial alignment for motion anchor
  TransformAlignment &initAlignment() { return _init_align; }

  int numDof() const { return _cfg.numDof(); }

private:
  BeyondMimicConfig _cfg;
  MotionCommand _command;
  std::vector<float> _last_action;
  TransformAlignment _init_align;

  // Joint mapping: robot order ↔ model (policy) order
  DoFAdapter _robot2model;   // robot dof_pos → model observation order
  DoFAdapter _model2robot;   // model action → robot pd_target order
};
