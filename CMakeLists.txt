cmake_minimum_required(VERSION 3.14)
project(rl_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime (复用 hl_motion 自带的)
set(ONNXRUNTIME_ROOT /workspace/hl_motion)
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS ${ONNXRUNTIME_ROOT}/lib
  NO_DEFAULT_PATH)

# ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# yaml-cpp + Eigen3
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${ONNXRUNTIME_ROOT}/include
  ${EIGEN3_INCLUDE_DIRS})

add_executable(rl_deploy_node
  src/main.cpp
  src/onnx_infer.cpp
  src/data_converter.cpp
  src/rl_controller_node.cpp)

ament_target_dependencies(rl_deploy_node
  rclcpp sensor_msgs geometry_msgs std_msgs)

target_link_libraries(rl_deploy_node
  ${ONNXRUNTIME_LIB}
  yaml-cpp)

install(TARGETS rl_deploy_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch model DESTINATION share/${PROJECT_NAME})

ament_package()
