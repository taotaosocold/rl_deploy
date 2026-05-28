#include "onnx_infer.hpp"
#include <sstream>
#include <cstring>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
std::vector<float> OnnxInfer::parseFloatList(const std::string &s) {
  std::vector<float> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(std::stof(item));
  }
  return out;
}

std::vector<std::string> OnnxInfer::parseStringList(const std::string &s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(item);
  }
  return out;
}

std::vector<const char *> OnnxInfer::inputNamePtrs() const {
  std::vector<const char *> ptrs;
  for (const auto &name : _input_names) ptrs.push_back(name.c_str());
  return ptrs;
}

std::vector<const char *> OnnxInfer::outputNamePtrs() const {
  std::vector<const char *> ptrs;
  for (const auto &name : _output_names) ptrs.push_back(name.c_str());
  return ptrs;
}

// ---------------------------------------------------------------------------
// Model loading + metadata parsing
// ---------------------------------------------------------------------------
bool OnnxInfer::loadModel(const std::string &model_path, bool parse_meta) {
  try {
    _opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    _session = std::make_unique<Ort::Session>(_env, model_path.c_str(), _opts);

    Ort::AllocatorWithDefaultOptions alloc;

    // --- input names & shapes ---
    size_t num_inputs = _session->GetInputCount();
    _input_names.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      auto name_ptr = _session->GetInputNameAllocated(i, alloc);
      _input_names[i] = name_ptr.get();
      auto shape = _session->GetInputTypeInfo(i)
                       .GetTensorTypeAndShapeInfo().GetShape();
      if (i == 0) {
        _obs_size = shape[1];
      }
      std::cout << "[OnnxInfer] Input  " << i << ": " << _input_names[i]
                << " shape=[";
      for (size_t j = 0; j < shape.size(); j++)
        std::cout << shape[j] << (j < shape.size() - 1 ? ", " : "");
      std::cout << "]" << std::endl;
    }

    // --- output names & shapes ---
    size_t num_outputs = _session->GetOutputCount();
    _output_names.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
      auto name_ptr = _session->GetOutputNameAllocated(i, alloc);
      _output_names[i] = name_ptr.get();
      auto shape = _session->GetOutputTypeInfo(i)
                       .GetTensorTypeAndShapeInfo().GetShape();
      if (i == 0) {
        _action_size = shape[1];
      }
      std::cout << "[OnnxInfer] Output " << i << ": " << _output_names[i]
                << " shape=[";
      for (size_t j = 0; j < shape.size(); j++)
        std::cout << shape[j] << (j < shape.size() - 1 ? ", " : "");
      std::cout << "]" << std::endl;
    }

    // --- parse metadata ---
    if (parse_meta) {
      parseModelMeta();
    }

    std::cout << "[OnnxInfer] Model loaded. obs=" << _obs_size
              << " action=" << _action_size
              << " dof=" << _meta.num_dof()
              << " bodies=" << _meta.num_bodies() << std::endl;
    return true;

  } catch (const Ort::Exception &e) {
    std::cerr << "[OnnxInfer] ERROR: " << e.what() << std::endl;
    return false;
  }
}

void OnnxInfer::parseModelMeta() {
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::ModelMetadata model_meta = _session->GetModelMetadata();

  auto keys = model_meta.GetCustomMetadataMapKeysAllocated(alloc);

  std::map<std::string, std::string> meta_map;
  for (const auto &key_ptr : keys) {
    std::string key(key_ptr.get());
    auto val_ptr = model_meta.LookupCustomMetadataMapAllocated(key.c_str(), alloc);
    std::string val(val_ptr.get());
    meta_map[key] = val;
    std::cout << "[OnnxInfer] metadata: " << key << std::endl;
  }

  auto get = [&](const std::string &k) -> std::string {
    auto it = meta_map.find(k);
    if (it != meta_map.end()) return it->second;
    return "";
  };

  std::string joint_names_str = get("joint_names");
  std::string default_pos_str = get("default_joint_pos");
  std::string stiffness_str  = get("joint_stiffness");
  std::string damping_str    = get("joint_damping");
  std::string action_scale_str = get("action_scale");
  std::string body_names_str = get("body_names");
  std::string anchor_str     = get("anchor_body_name");

  if (!joint_names_str.empty()) {
    _meta.joint_names = parseStringList(joint_names_str);
    std::cout << "[OnnxInfer] Parsed " << _meta.joint_names.size()
              << " joint names from model metadata" << std::endl;
  }
  if (!default_pos_str.empty()) {
    _meta.default_joint_pos = parseFloatList(default_pos_str);
  }
  if (!stiffness_str.empty()) {
    _meta.stiffness = parseFloatList(stiffness_str);
  }
  if (!damping_str.empty()) {
    _meta.damping = parseFloatList(damping_str);
  }
  if (!action_scale_str.empty()) {
    _meta.action_scales = parseFloatList(action_scale_str);
  }
  if (!body_names_str.empty()) {
    _meta.body_names = parseStringList(body_names_str);
  }
  if (!anchor_str.empty()) {
    _meta.anchor_body_name = anchor_str;
  }

  std::cout << "[OnnxInfer] Metadata summary: joints=" << _meta.num_dof()
            << " bodies=" << _meta.num_bodies()
            << " anchor=" << _meta.anchor_body_name << std::endl;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
InferenceResult OnnxInfer::infer(const std::vector<float> &observation,
                                  int64_t timestep) {
  InferenceResult result;

  // --- build inputs ---
  std::vector<Ort::Value> inputs;
  std::vector<int64_t> obs_shape = {1, _obs_size};
  inputs.push_back(Ort::Value::CreateTensor<float>(
      _memory,
      const_cast<float *>(observation.data()),
      observation.size(),
      obs_shape.data(),
      obs_shape.size()));

  std::vector<int64_t> ts_shape = {1, 1};
  std::vector<float> ts_data = {static_cast<float>(timestep)};
  inputs.push_back(Ort::Value::CreateTensor<float>(
      _memory,
      ts_data.data(),
      ts_data.size(),
      ts_shape.data(),
      ts_shape.size()));

  // --- run ---
  auto in_ptrs = inputNamePtrs();
  auto out_ptrs = outputNamePtrs();

  auto outputs = _session->Run(
      Ort::RunOptions{nullptr},
      in_ptrs.data(), inputs.data(), inputs.size(),
      out_ptrs.data(), out_ptrs.size());

  // --- extract outputs ---
  if (outputs.size() > 0 && outputs[0].IsTensor()) {
    float *data = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = shape[1];
    result.actions.assign(data, data + count);
  }

  if (outputs.size() > 1 && outputs[1].IsTensor()) {
    float *data = outputs[1].GetTensorMutableData<float>();
    auto shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = shape[1];
    result.joint_pos.assign(data, data + count);
  }

  if (outputs.size() > 2 && outputs[2].IsTensor()) {
    float *data = outputs[2].GetTensorMutableData<float>();
    auto shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = shape[1];
    result.joint_vel.assign(data, data + count);
  }

  if (outputs.size() > 3 && outputs[3].IsTensor()) {
    float *data = outputs[3].GetTensorMutableData<float>();
    auto shape = outputs[3].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = shape[1] * shape[2];
    result.body_pos_w.assign(data, data + count);
  }

  if (outputs.size() > 4 && outputs[4].IsTensor()) {
    float *data = outputs[4].GetTensorMutableData<float>();
    auto shape = outputs[4].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = shape[1] * shape[2];
    result.body_quat_w.assign(data, data + count);
  }

  return result;
}

InferenceResult OnnxInfer::dryRun(int64_t start_timestep) {
  std::vector<float> zeros(_obs_size, 0.0f);
  return infer(zeros, start_timestep);
}
