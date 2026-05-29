#!/usr/bin/env python3
"""
Read-only ROS2 communication test for CASBOT02.
Subscribes to /motion/joint_state and /motion/imu, prints stats.
Publishes NOTHING — safe to run anytime, robot won't move.

Usage:
  python3 scripts/check_state.py
  python3 scripts/check_state.py --rate 0.5   # print every 2 seconds
"""

import argparse
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu


class StateChecker(Node):
    def __init__(self, print_rate=1.0):
        super().__init__("state_checker")

        self._js_count = 0
        self._imu_count = 0
        self._js_msg = None
        self._imu_msg = None
        self._start_time = time.time()

        self._js_sub = self.create_subscription(
            JointState, "/motion/joint_state", self._js_cb, 10
        )
        self._imu_sub = self.create_subscription(
            Imu, "/motion/imu", self._imu_cb, 10
        )

        self._timer = self.create_timer(1.0 / print_rate, self._print_stats)
        self.get_logger().info("State checker started (read-only). Waiting for data...")

    def _js_cb(self, msg):
        if self._js_count == 0:
            self.get_logger().info(
                f"First joint_state: {len(msg.name)} joints"
            )
            # Print joint names in received order
            for i, name in enumerate(msg.name):
                self.get_logger().info(f"  [{i:2d}] {name}")
        self._js_msg = msg
        self._js_count += 1

    def _imu_cb(self, msg):
        if self._imu_count == 0:
            self.get_logger().info("First IMU message received")
        self._imu_msg = msg
        self._imu_count += 1

    def _print_stats(self):
        elapsed = time.time() - self._start_time
        js_hz = self._js_count / elapsed if elapsed > 0 else 0
        imu_hz = self._imu_count / elapsed if elapsed > 0 else 0

        status = []
        status.append(f"uptime={elapsed:.0f}s")
        status.append(f"joint_state: {self._js_count} msgs, {js_hz:.1f} Hz")
        status.append(f"imu:         {self._imu_count} msgs, {imu_hz:.1f} Hz")

        if self._js_msg:
            # Show a few key joint positions
            key = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                   "left_elbow_pitch_joint", "right_elbow_pitch_joint",
                   "head_yaw_joint", "head_pitch_joint"]
            vals = []
            for i, name in enumerate(self._js_msg.name):
                if name in key:
                    pos = self._js_msg.position[i] if i < len(self._js_msg.position) else 0
                    vals.append(f"{name}={pos:.3f}")
            if vals:
                status.append(" | ".join(vals))

        if self._imu_msg:
            q = self._imu_msg.orientation
            status.append(
                f"imu_quat(wxyz)=[{q.w:.3f}, {q.x:.3f}, {q.y:.3f}, {q.z:.3f}]"
            )

        self.get_logger().info(" | ".join(status))


def main():
    parser = argparse.ArgumentParser(description="Read-only ROS2 state checker")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Print interval in seconds (default: 1.0)")
    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    node = StateChecker(print_rate=args.rate)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
