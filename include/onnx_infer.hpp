#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <iostream>

/// Parsed from ONNX model custom_metadata_map
struct ModelMeta {
  std::vector<std::string> joint_names;
  std::vector<float> default_joint_pos;
  std::vector<float> stiffness;
  std::vector<float> damping;
  std::vector<float> action_scales;
  std::string anchor_body_name;
  std::vector<std::string> body_names;

  int num_dof() const { return static_cast<int>(joint_names.size()); }
  int num_bodies() const { return static_cast<int>(body_names.size()); }
  bool valid() const { return num_dof() > 0; }
};

/// All outputs from one BeyondMimic ONNX inference call
struct InferenceResult {
  std::vector<float> actions;       // [num_dof]
  std::vector<float> joint_pos;     // [num_dof] reference motion
  std::vector<float> joint_vel;     // [num_dof] reference motion
  std::vector<float> body_pos_w;    // [num_bodies * 3] flattened
  std::vector<float> body_quat_w;   // [num_bodies * 4] flattened (w,x,y,z order)
};

class OnnxInfer {
public:
  OnnxInfer() = default;
  ~OnnxInfer() = default;

  /// Load ONNX model and parse metadata.
  bool loadModel(const std::string &model_path, bool parse_meta = true);

  /// Run inference with observation and timestep.
  InferenceResult infer(const std::vector<float> &observation, int64_t timestep);

  /// Dry-run to warm up and initialize internal state.
  InferenceResult dryRun(int64_t start_timestep = 0);

  // --- accessors ---
  int64_t obsSize()     const { return _obs_size; }
  int64_t actionSize()  const { return _action_size; }
  int numDof()          const { return _meta.num_dof(); }
  int numBodies()       const { return _meta.num_bodies(); }

  const ModelMeta &modelMeta() const { return _meta; }

private:
  void parseModelMeta();
  static std::vector<float> parseFloatList(const std::string &s);
  static std::vector<std::string> parseStringList(const std::string &s);

  // Build const char* arrays from stored strings (for ONNX API)
  std::vector<const char *> inputNamePtrs() const;
  std::vector<const char *> outputNamePtrs() const;

  Ort::Env _env{ORT_LOGGING_LEVEL_WARNING, "rl_deploy"};
  Ort::SessionOptions _opts;
  std::unique_ptr<Ort::Session> _session;

  std::vector<std::string> _input_names;
  std::vector<std::string> _output_names;
  int64_t _obs_size = 0;
  int64_t _action_size = 0;

  ModelMeta _meta;

  Ort::MemoryInfo _memory{Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)};
};
